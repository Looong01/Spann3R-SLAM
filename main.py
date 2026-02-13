"""
Main script for Spann3R-SLAM.
"""

import argparse
import datetime
import pathlib
import time
import warnings
import cv2
import lietorch
import torch
import tqdm
import yaml
import sys

# Suppress known safe warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*The parameter 'pretrained' is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*Arguments other than a weight enum.*", category=UserWarning
)
from spann3r_slam.global_opt import FactorGraph

from spann3r_slam.config import load_config, config, set_global_config
from spann3r_slam.dataloader import Intrinsics, load_dataset
import spann3r_slam.evaluate as eval
from spann3r_slam.frame import (
    Mode,
    SharedKeyframes,
    SharedStates,
    create_frame,
)
from spann3r_slam.spann3r_utils import (
    load_spann3r,
    load_spann3r_retriever,
    spann3r_inference_mono,
    spann3r_render,
)
from spann3r_slam.multiprocess_utils import new_queue, try_get_msg
from spann3r_slam.tracker import FrameTracker
from spann3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(
    cfg, states, keyframes, K, checkpoint_path, dust3r_checkpoint_path, device
):
    set_global_config(cfg)

    print("[Backend] Loading Spann3R model in backend process...")
    model = load_spann3r(
        path=checkpoint_path,
        device=device,
        dust3r_path=dust3r_checkpoint_path,
    )
    if K is not None:
        K = K.to(device=device, dtype=torch.float32)

    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_spann3r_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/examples/s00567")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/spann3r.pth",
        help="Path to Spann3R checkpoint (default: checkpoints/spann3r.pth)",
    )
    parser.add_argument(
        "--dust3r-checkpoint",
        default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="Path to DUSt3R checkpoint (default: checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth)",
    )
    parser.add_argument(
        "--no-render-gaussians",
        action="store_true",
        help="Disable Spann3R backend rendering and per-frame PNG saving",
    )
    parser.add_argument(
        "--render-dir",
        default="logs/spann3r_renders",
        help="Directory to save Spann3R-rendered images (default: logs/spann3r_renders)",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=4 * 1024 * 1024,
        help="Max number of points/gaussians used by Spann3R renderer (default: 4194304)",
    )
    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=4,
        help="Spatial stride for subsampling Gaussians per frame (default: 4, stride=1 means no subsampling)",
    )

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(
                config,
                states,
                keyframes,
                main2viz,
                viz2main,
                args.spatial_stride,
                args.max_gaussians,
            ),
        )
        viz.start()

    # Load Spann3R model
    print("Loading Spann3R model...")
    model = load_spann3r(
        path=args.checkpoint,
        device=device,
        dust3r_path=args.dust3r_checkpoint,
    )
    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)
    K_backend = K.detach().cpu() if K is not None else None

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)

    # Spann3R rendering setup
    render_gaussians = not args.no_render_gaussians
    spatial_stride = args.spatial_stride
    max_gaussians = args.max_gaussians
    last_msg = WindowMsg(spatial_stride=spatial_stride, max_gaussians=max_gaussians)
    render_dir = None
    if render_gaussians:
        render_dir = pathlib.Path(args.render_dir)
        render_dir.mkdir(exist_ok=True, parents=True)
        print(f"[Spann3R Rendering] Enabled. Saving to {render_dir}")
    else:
        print("[Spann3R Rendering] Disabled.")

    backend = mp.Process(
        target=run_backend,
        args=(
            config,
            states,
            keyframes,
            K_backend,
            args.checkpoint,
            args.dust3r_checkpoint,
            device,
        ),
    )
    backend.start()

    def render_and_save_frame(src_frame, ref_frame, frame_idx):
        if not render_gaussians:
            return
        stride = max(1, int(last_msg.spatial_stride))
        max_points = max(20000, int(last_msg.max_gaussians))

        rendered = spann3r_render(
            model,
            src_frame,
            ref_frame,
            K=K,
            target_T_WC=src_frame.T_WC,
            spatial_stride=stride,
            max_points=max_points,
        )
        if rendered is None:
            return

        rendered_img = rendered[0, 0].detach().cpu().clamp(0, 1).permute(1, 2, 0)
        rendered_np = (rendered_img.numpy() * 255).astype("uint8")
        rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(render_dir / f"spann3r_{frame_idx:06d}.png"), rendered_bgr)

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        add_new_kf = False

        if mode == Mode.INIT:
            # Initialize via mono inference with Spann3R
            X_init, C_init = spann3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            render_and_save_frame(frame, frame, i)

            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            if not try_reloc:
                keyframe = keyframes.last_keyframe()
                if keyframe is None:
                    keyframe = frame
                render_and_save_frame(frame, keyframe, i)

        elif mode == Mode.RELOC:
            X, C = spann3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            render_and_save_frame(frame, frame, i)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
