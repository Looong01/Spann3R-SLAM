import dataclasses
import weakref
from pathlib import Path

import imgui
import lietorch
import moderngl
import moderngl_window as mglw
import numpy as np
import torch
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.image import Image
from in3d.pose_utils import translation_matrix
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from spann3r_slam.config import config, set_global_config
from spann3r_slam.lietorch_utils import as_SE3
from spann3r_slam.spann3r_utils import spann3r_collect_world_points
from spann3r_slam.visualization_utils import Frustums, Lines, image_with_text


_CV2GL = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
)


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 0.0
    render_resolution_scale: float = 0.5
    spatial_stride: int = 4
    max_gaussians: int = 4 * 1024 * 1024
    render_point_radius: int = 1
    render_refresh_interval: int = 1


class Window(WindowEvents):
    title = "Spann3R-SLAM"
    window_size = (1960, 1080)

    def __init__(
        self,
        states,
        keyframes,
        main2viz,
        viz2main,
        init_spatial_stride=4,
        init_max_gaussians=4 * 1024 * 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"

        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2.0

        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())
        self.line_prog = self.load_program("programs/lines.glsl")

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )

        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)
        self.viewport = ViewportWindow("Scene", self.camera)

        self.state = WindowMsg()
        self.state.C_conf_threshold = float(config.get("tracking", {}).get("C_conf", 0.0))
        self.state.spatial_stride = max(1, int(init_spatial_stride))
        self.state.max_gaussians = max(20000, int(init_max_gaussians))
        self.states = states
        self.keyframes = keyframes
        self.main2viz = main2viz
        self.viz2main = viz2main

        self.show_keyframe_edges = True
        self.follow_cam = True
        self.show_keyframe = True
        self.show_axis = True
        self.line_thickness = 3.0
        self.frustum_scale = 0.05
        self.culling = True

        self.curr_img = Image()
        self.kf_img = Image()
        self.gs_render_img = Image()

        self.use_spann3r_rendering = True
        self.gs_tex = None
        self.gs_depth_tex = None
        self.gs_quad_prog = self.ctx.program(
            vertex_shader="""
            #version 330 core
            out vec2 uv;
            void main() {
                float x = float(gl_VertexID % 2) * 2.0 - 1.0;
                float y = float(gl_VertexID / 2) * 2.0 - 1.0;
                gl_Position = vec4(x, y, 0.0, 1.0);
                uv = vec2((x + 1.0) * 0.5, (-y + 1.0) * 0.5);
            }
            """,
            fragment_shader="""
            #version 330 core
            uniform sampler2D gs_texture;
            uniform sampler2D gs_depth;
            in vec2 uv;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(texture(gs_texture, uv).rgb, 1.0);
                gl_FragDepth = texture(gs_depth, uv).r;
            }
            """,
        )
        self.gs_quad_vao = self.ctx.vertex_array(self.gs_quad_prog, [])

        # Persistent world cloud cache to avoid losing geometry outside the latest view.
        self._kf_points = np.zeros((0, 3), dtype=np.float32)
        self._kf_colors = np.zeros((0, 3), dtype=np.float32)
        self._curr_points = np.zeros((0, 3), dtype=np.float32)
        self._curr_colors = np.zeros((0, 3), dtype=np.float32)
        self._cached_kf_count = 0
        self._cache_signature = None
        self._last_curr_frame_id = -1
        self._render_counter = 0
        self._active_render_points = 0
        self._offset_cache = {0: np.array([[0, 0]], dtype=np.int32)}

    def render(self, t: float, frametime: float):
        del t, frametime
        self._render_counter += 1
        self.viewport.use()

        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        curr_frame = self.states.get_frame()
        h, w = [int(x) for x in curr_frame.img_shape.flatten().tolist()]
        if h <= 0 or w <= 0:
            h = int(self.states.h)
            w = int(self.states.w)
        self.frustums.make_frustum(h, w)
        self.curr_img.write(curr_frame.uimg.numpy())

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        if self.follow_cam:
            T_WC = cam_T_WC.matrix().numpy().astype(np.float32) @ translation_matrix(
                np.array([0, 0, -2], dtype=np.float32)
            )
            self.camera.follow_cam(np.linalg.inv(T_WC))
        else:
            self.camera.unfollow_cam()

        with self.keyframes.lock:
            n_keyframes = len(self.keyframes)

        self._update_world_cloud_cache(curr_frame, n_keyframes)

        gs_render = self._render_gs_interactive() if self.use_spann3r_rendering else None
        if gs_render is not None:
            gs_img_np, gs_depth_np = gs_render
        else:
            gs_img_np = self.states.get_gs_rendered()
            gs_depth_np = None

        if gs_img_np is not None:
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.depth_func = "<="
            self.ctx.disable(moderngl.CULL_FACE)
            self._render_gs_fullscreen(gs_img_np, gs_depth_np)
            self.gs_render_img.write(gs_img_np)
            if self.culling:
                self.ctx.enable(moderngl.CULL_FACE)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )

        for kf_idx in range(n_keyframes):
            keyframe = self.keyframes[kf_idx]
            if kf_idx == n_keyframes - 1:
                self.kf_img.write(keyframe.uimg.numpy())

            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=[1, 0, 0, 1],
                    thickness=self.line_thickness * self.scale,
                )

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()

        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)

        new_state = dataclasses.replace(self.state)
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0.0, 5.0
        )

        imgui.spacing()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)
        _, self.use_spann3r_rendering = imgui.checkbox(
            "spann3r_rendering", self.use_spann3r_rendering
        )

        imgui.spacing()
        imgui.text("Render Tuning")
        _, new_state.render_resolution_scale = imgui.slider_float(
            "render_res_scale", self.state.render_resolution_scale, 0.2, 1.0
        )
        _, new_state.spatial_stride = imgui.slider_int(
            "spatial_stride", self.state.spatial_stride, 1, 16
        )
        max_slider = max(8 * 1024 * 1024, int(self.state.max_gaussians))
        _, new_state.max_gaussians = imgui.slider_int(
            "max_gaussians", self.state.max_gaussians, 20000, max_slider
        )
        _, new_state.render_point_radius = imgui.slider_int(
            "render_point_radius", self.state.render_point_radius, 0, 2
        )
        _, new_state.render_refresh_interval = imgui.slider_int(
            "cache_refresh", self.state.render_refresh_interval, 1, 30
        )

        imgui.text(f"active points: {self._active_render_points}")
        imgui.text(f"cached keyframes: {self._cached_kf_count}")

        imgui.spacing()
        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )
        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0.0, 0.1
        )

        imgui.spacing()
        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        image_with_text(self.gs_render_img, size, "spann3r", same_line=False)
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)
        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def _cloud_signature(self):
        return (
            float(self.state.C_conf_threshold),
            int(self.state.spatial_stride),
            int(self.state.max_gaussians),
        )

    def _reset_cloud_cache(self):
        self._kf_points = np.zeros((0, 3), dtype=np.float32)
        self._kf_colors = np.zeros((0, 3), dtype=np.float32)
        self._curr_points = np.zeros((0, 3), dtype=np.float32)
        self._curr_colors = np.zeros((0, 3), dtype=np.float32)
        self._cached_kf_count = 0
        self._last_curr_frame_id = -1

    def _frame_world_cloud_np(self, frame):
        pts, rgb = spann3r_collect_world_points(
            [frame],
            conf_thresh=float(self.state.C_conf_threshold),
            spatial_stride=int(self.state.spatial_stride),
            max_points=None,
        )
        if pts is None:
            return None, None
        return (
            pts.detach().cpu().numpy().astype(np.float32),
            rgb.detach().cpu().numpy().astype(np.float32),
        )

    @staticmethod
    def _cap_cloud_np(points, colors, max_points):
        if points.shape[0] <= max_points:
            return points, colors
        idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        return points[idx], colors[idx]

    def _append_to_kf_cache(self, points, colors):
        if points is None or points.shape[0] == 0:
            return
        if self._kf_points.shape[0] == 0:
            self._kf_points = points
            self._kf_colors = colors
        else:
            self._kf_points = np.concatenate([self._kf_points, points], axis=0)
            self._kf_colors = np.concatenate([self._kf_colors, colors], axis=0)

        max_points = max(20000, int(self.state.max_gaussians))
        self._kf_points, self._kf_colors = self._cap_cloud_np(
            self._kf_points, self._kf_colors, max_points
        )

    def _update_world_cloud_cache(self, curr_frame, n_keyframes):
        signature = self._cloud_signature()
        if signature != self._cache_signature or n_keyframes < self._cached_kf_count:
            self._cache_signature = signature
            self._reset_cloud_cache()

        if self._cached_kf_count < n_keyframes:
            for kf_idx in range(self._cached_kf_count, n_keyframes):
                kf = self.keyframes[kf_idx]
                pts, rgb = self._frame_world_cloud_np(kf)
                self._append_to_kf_cache(pts, rgb)
            self._cached_kf_count = n_keyframes

        refresh_interval = max(1, int(self.state.render_refresh_interval))
        should_refresh_curr = (
            curr_frame.frame_id != self._last_curr_frame_id
            or (self._render_counter % refresh_interval) == 0
        )
        if should_refresh_curr:
            pts, rgb = self._frame_world_cloud_np(curr_frame)
            self._curr_points = np.zeros((0, 3), dtype=np.float32) if pts is None else pts
            self._curr_colors = np.zeros((0, 3), dtype=np.float32) if rgb is None else rgb
            self._last_curr_frame_id = int(curr_frame.frame_id)

    def _compose_render_cloud(self):
        parts_p = []
        parts_c = []
        if self._kf_points.shape[0] > 0:
            parts_p.append(self._kf_points)
            parts_c.append(self._kf_colors)
        if self._curr_points.shape[0] > 0:
            parts_p.append(self._curr_points)
            parts_c.append(self._curr_colors)

        if not parts_p:
            self._active_render_points = 0
            return None, None

        points = np.concatenate(parts_p, axis=0)
        colors = np.concatenate(parts_c, axis=0)
        max_points = max(20000, int(self.state.max_gaussians))
        points, colors = self._cap_cloud_np(points, colors, max_points)
        self._active_render_points = int(points.shape[0])
        return points, colors

    def _render_gs_fullscreen(self, gs_img_np: np.ndarray, gs_depth_np: np.ndarray | None):
        h, w = gs_img_np.shape[:2]
        if self.gs_tex is None or self.gs_tex.size != (w, h):
            if self.gs_tex is not None:
                self.gs_tex.release()
            self.gs_tex = self.ctx.texture((w, h), 3, dtype="f4")
            self.gs_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        if self.gs_depth_tex is None or self.gs_depth_tex.size != (w, h):
            if self.gs_depth_tex is not None:
                self.gs_depth_tex.release()
            self.gs_depth_tex = self.ctx.texture((w, h), 1, dtype="f4")
            self.gs_depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        if gs_depth_np is None:
            gs_depth_np = np.ones((h, w), dtype=np.float32)

        self.gs_tex.write(gs_img_np.astype(np.float32).tobytes())
        self.gs_depth_tex.write(gs_depth_np.astype(np.float32).tobytes())
        self.gs_tex.use(0)
        self.gs_depth_tex.use(1)
        self.gs_quad_prog["gs_texture"].value = 0
        self.gs_quad_prog["gs_depth"].value = 1
        self.gs_quad_vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)

    def _camera_T_CW_cv(self) -> np.ndarray:
        T_CW_gl = self.camera.T_CW.astype(np.float32)
        return _CV2GL @ T_CW_gl

    def _camera_intrinsics(self, w: int, h: int) -> np.ndarray:
        P = self.camera.proj_mat.matrix.astype(np.float32)
        fx = abs(P[0, 0]) * 0.5 * float(w)
        fy = abs(P[1, 1]) * 0.5 * float(h)
        if (not np.isfinite(fx)) or (not np.isfinite(fy)) or fx < 1e-6 or fy < 1e-6:
            focal = 0.5 * max(h, w) / np.tan(np.deg2rad(60.0 / 2.0))
            fx, fy = focal, focal
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def _pixel_offsets(self, radius: int) -> np.ndarray:
        radius = max(0, int(radius))
        if radius in self._offset_cache:
            return self._offset_cache[radius]

        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append((dx, dy))
        self._offset_cache[radius] = np.asarray(offsets, dtype=np.int32)
        return self._offset_cache[radius]

    def _depth_from_cv_z(self, z: np.ndarray) -> np.ndarray:
        znear = float(self.camera.proj_mat.znear)
        zfar = float(self.camera.proj_mat.zfar)
        z = np.maximum(z, 1e-6)
        ndc_z = ((zfar + znear) / (zfar - znear)) - ((2.0 * zfar * znear) / ((zfar - znear) * z))
        depth = 0.5 * ndc_z + 0.5
        return np.clip(depth, 0.0, 1.0).astype(np.float32)

    def _render_gs_interactive(self):
        world_pts, world_rgb = self._compose_render_cloud()
        if world_pts is None:
            return None

        view_w, view_h = self.viewport.screen.size
        scale = float(np.clip(self.state.render_resolution_scale, 0.2, 1.0))
        w = max(32, int(view_w * scale))
        h = max(32, int(view_h * scale))

        T_CW = self._camera_T_CW_cv()
        R = T_CW[:3, :3]
        t = T_CW[:3, 3]

        pts_cam = world_pts @ R.T + t[None]
        z = pts_cam[:, 2]
        valid = np.isfinite(pts_cam).all(axis=1) & (z > 1e-6)
        if not np.any(valid):
            return None

        pts_cam = pts_cam[valid]
        colors = world_rgb[valid]
        z = pts_cam[:, 2]

        K = self._camera_intrinsics(w, h)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = np.rint(fx * (pts_cam[:, 0] / z) + cx).astype(np.int32)
        v = np.rint(fy * (pts_cam[:, 1] / z) + cy).astype(np.int32)

        radius = int(self.state.render_point_radius)
        if radius > 0:
            offsets = self._pixel_offsets(radius)
            u = u[:, None] + offsets[None, :, 0]
            v = v[:, None] + offsets[None, :, 1]
            z = np.repeat(z[:, None], offsets.shape[0], axis=1)
            colors = np.repeat(colors[:, None, :], offsets.shape[0], axis=1)
            u = u.reshape(-1)
            v = v.reshape(-1)
            z = z.reshape(-1)
            colors = colors.reshape(-1, 3)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(valid):
            return None

        u = u[valid]
        v = v[valid]
        z = z[valid]
        colors = colors[valid]

        pix = v * w + u
        order = np.lexsort((z, pix))
        pix_sorted = pix[order]
        z_sorted = z[order]
        color_sorted = colors[order]
        unique_pix, first_idx = np.unique(pix_sorted, return_index=True)

        img = np.zeros((h * w, 3), dtype=np.float32)
        img[unique_pix] = color_sorted[first_idx]

        z_buffer = np.full(h * w, np.inf, dtype=np.float32)
        z_buffer[unique_pix] = z_sorted[first_idx]
        depth = np.ones(h * w, dtype=np.float32)
        valid_depth = np.isfinite(z_buffer)
        depth[valid_depth] = self._depth_from_cv_z(z_buffer[valid_depth])

        return img.reshape(h, w, 3), depth.reshape(h, w)


def run_visualization(
    cfg,
    states,
    keyframes,
    main2viz,
    viz2main,
    init_spatial_stride=4,
    init_max_gaussians=4 * 1024 * 1024,
) -> None:
    set_global_config(cfg)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        keyframes=keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        init_spatial_stride=init_spatial_stride,
        init_max_gaussians=init_max_gaussians,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )

    window._config = weakref.ref(window_config)
    window.swap_buffers()
    window.set_default_viewport()
    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()

        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)

        window.use()
        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
