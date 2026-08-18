from __future__ import annotations

_VERTEX_SHADER = r"""
#version 330

in vec2 in_corner;

uniform sampler2D u_scene_data;
uniform sampler2D u_sh_data;
uniform isampler2D u_draw_order;
uniform int u_scene_texture_width;
uniform int u_sh_texture_width;
uniform int u_sh_coefficient_count;
uniform int u_sh_degree;
uniform int u_order_texture_width;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec2 u_viewport;
uniform vec2 u_focal;
uniform float u_near;
uniform float u_far;

uniform vec3 u_up;
uniform float u_height_min;
uniform float u_height_max;
uniform float u_scan_reverse;
uniform float u_scan_progress;
uniform float u_scan_feather;
uniform float u_rep_source;
uniform float u_rep_target;
uniform int u_transition_effect;
uniform vec3 u_scene_center;
uniform float u_scene_radius;
uniform vec3 u_camera_position;

uniform float u_point_radius;
uniform float u_point_opacity;
uniform float u_splat_scale;
uniform float u_splat_opacity;
uniform float u_min_splat_pixels;
uniform float u_max_splat_pixels;
uniform float u_pixel_scale;

out vec2 v_local;
out vec3 v_color;
out float v_opacity;
out float v_depth;

vec4 fetch_scene(int linear_index) {
    ivec2 location = ivec2(
        linear_index % u_scene_texture_width,
        linear_index / u_scene_texture_width
    );
    return texelFetch(u_scene_data, location, 0);
}

vec3 fetch_sh(int point_index, int coefficient_index) {
    int linear_index = point_index * u_sh_coefficient_count + coefficient_index;
    ivec2 location = ivec2(
        linear_index % u_sh_texture_width,
        linear_index / u_sh_texture_width
    );
    return texelFetch(u_sh_data, location, 0).rgb;
}

vec3 evaluate_sh_color(int point_index, vec3 position, vec3 dc_color) {
    if (u_sh_degree == 0) {
        return dc_color;
    }

    vec3 direction = position - u_camera_position;
    direction /= max(length(direction), 1e-7);
    float x = direction.x;
    float y = direction.y;
    float z = direction.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    vec3 color = dc_color
        - 0.4886025119029199 * y * fetch_sh(point_index, 0)
        + 0.4886025119029199 * z * fetch_sh(point_index, 1)
        - 0.4886025119029199 * x * fetch_sh(point_index, 2);
    if (u_sh_degree > 1) {
        color +=
            1.0925484305920792 * x * y * fetch_sh(point_index, 3)
            - 1.0925484305920792 * y * z * fetch_sh(point_index, 4)
            + 0.31539156525252005 * (2.0 * zz - xx - yy)
                * fetch_sh(point_index, 5)
            - 1.0925484305920792 * x * z * fetch_sh(point_index, 6)
            + 0.5462742152960396 * (xx - yy) * fetch_sh(point_index, 7);
    }
    if (u_sh_degree > 2) {
        color +=
            -0.5900435899266435 * y * (3.0 * xx - yy)
                * fetch_sh(point_index, 8)
            + 2.890611442640554 * x * y * z * fetch_sh(point_index, 9)
            - 0.4570457994644658 * y * (4.0 * zz - xx - yy)
                * fetch_sh(point_index, 10)
            + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                * fetch_sh(point_index, 11)
            - 0.4570457994644658 * x * (4.0 * zz - xx - yy)
                * fetch_sh(point_index, 12)
            + 1.445305721320277 * z * (xx - yy) * fetch_sh(point_index, 13)
            - 0.5900435899266435 * x * (xx - 3.0 * yy)
                * fetch_sh(point_index, 14);
    }
    return max(color, vec3(0.0));
}

int fetch_draw_index(int linear_index) {
    ivec2 location = ivec2(
        linear_index % u_order_texture_width,
        linear_index / u_order_texture_width
    );
    return texelFetch(u_draw_order, location, 0).r;
}

void main() {
    int point_index = fetch_draw_index(gl_InstanceID);
    vec4 position_opacity = fetch_scene(point_index * 4);
    vec4 color_data = fetch_scene(point_index * 4 + 1);
    vec4 covariance_a_data = fetch_scene(point_index * 4 + 2);
    vec4 covariance_b_data = fetch_scene(point_index * 4 + 3);
    vec3 in_position = position_opacity.xyz;
    float in_opacity = position_opacity.w;
    vec3 in_color = evaluate_sh_color(point_index, in_position, color_data.xyz);
    vec3 in_covariance_a = covariance_a_data.xyz;
    vec3 in_covariance_b = covariance_b_data.xyz;

    vec4 camera_h = u_view * u_model * vec4(in_position, 1.0);
    vec3 camera = camera_h.xyz;
    float depth = -camera.z;
    vec4 clip = u_projection * camera_h;

    float height_range = max(u_height_max - u_height_min, 1e-7);
    float height = dot(in_position, u_up);
    float scan_coordinate = clamp((u_height_max - height) / height_range, 0.0, 1.0);
    float transition_coordinate = scan_coordinate;
    if (u_transition_effect != 0) {
        vec3 normalized_position =
            (in_position - u_scene_center) / max(u_scene_radius, 1e-7);
        if (u_transition_effect == 1) {
            // A spherical reveal feels natural for isolated objects and statues.
            transition_coordinate = clamp(length(normalized_position), 0.0, 1.0);
        } else if (u_transition_effect == 4) {
            // Stable object-space noise: individual splats never flicker over time.
            transition_coordinate = fract(sin(dot(
                normalized_position,
                vec3(127.1, 311.7, 74.7)
            )) * 43758.5453);
        } else {
            vec3 reference = abs(u_up.x) < 0.9
                ? vec3(1.0, 0.0, 0.0)
                : vec3(0.0, 0.0, 1.0);
            vec3 horizontal_a = normalize(reference - u_up * dot(reference, u_up));
            vec3 horizontal_b = cross(u_up, horizontal_a);
            float horizontal_x = dot(normalized_position, horizontal_a);
            float horizontal_y = dot(normalized_position, horizontal_b);
            if (u_transition_effect == 2) {
                // Two low frequencies avoid a mechanically straight edge.
                transition_coordinate = clamp(
                    scan_coordinate +
                    0.075 * sin(horizontal_x * 11.0 + horizontal_y * 3.0) +
                    0.035 * sin(horizontal_y * 17.0 - horizontal_x * 2.0),
                    0.0,
                    1.0
                );
            } else {
                float angle = atan(horizontal_y, horizontal_x) / 6.28318530718 + 0.5;
                transition_coordinate = fract(angle + scan_coordinate * 0.72);
            }
        }
    }
    if (u_scan_reverse > 0.5) {
        transition_coordinate = 1.0 - transition_coordinate;
    }
    float feather = max(u_scan_feather, 0.0001);
    float wave = u_scan_progress * (1.0 + 2.0 * feather) - feather;
    float local_progress = smoothstep(
        transition_coordinate - feather,
        transition_coordinate + feather,
        wave
    );
    float representation = mix(u_rep_source, u_rep_target, local_progress);

    float point_sigma = max(u_point_radius / 2.5, 0.15 * u_pixel_scale);
    vec2 pixel_offset = in_corner * point_sigma;

    if (depth <= u_near || depth >= u_far || clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_opacity = 0.0;
    } else {
        // Pure point-cloud frames skip all covariance projection work. The
        // branch is coherent for the whole draw outside an active transition.
        if (representation > 0.0001) {
            mat3 covariance = mat3(
                vec3(in_covariance_a.x, in_covariance_a.y, in_covariance_a.z),
                vec3(in_covariance_a.y, in_covariance_b.x, in_covariance_b.y),
                vec3(in_covariance_a.z, in_covariance_b.y, in_covariance_b.z)
            );
            mat3 camera_rotation = mat3(u_view * u_model);
            mat3 camera_covariance =
                camera_rotation * covariance * transpose(camera_rotation);

            float safe_depth = max(depth, u_near);
            vec3 jacobian_x = vec3(
                u_focal.x / safe_depth,
                0.0,
                u_focal.x * camera.x / (safe_depth * safe_depth)
            );
            vec3 jacobian_y = vec3(
                0.0,
                u_focal.y / safe_depth,
                u_focal.y * camera.y / (safe_depth * safe_depth)
            );
            float cov_xx = dot(jacobian_x, camera_covariance * jacobian_x);
            float cov_xy = dot(jacobian_x, camera_covariance * jacobian_y);
            float cov_yy = dot(jacobian_y, camera_covariance * jacobian_y);

            float antialias_variance = 0.09 * u_pixel_scale * u_pixel_scale;
            cov_xx = max(
                cov_xx * u_splat_scale * u_splat_scale + antialias_variance,
                0.0
            );
            cov_xy = cov_xy * u_splat_scale * u_splat_scale;
            cov_yy = max(
                cov_yy * u_splat_scale * u_splat_scale + antialias_variance,
                0.0
            );
            float trace = cov_xx + cov_yy;
            float discriminant = sqrt(max(
                (cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy,
                0.0
            ));
            float lambda_major = max(0.5 * (trace + discriminant), 0.0);
            float lambda_minor = max(0.5 * (trace - discriminant), 0.0);
            float sigma_major = clamp(
                sqrt(lambda_major),
                u_min_splat_pixels,
                u_max_splat_pixels
            );
            float sigma_minor = clamp(
                sqrt(lambda_minor),
                u_min_splat_pixels,
                u_max_splat_pixels
            );

            vec2 major_axis;
            if (abs(cov_xy) > 0.00001) {
                major_axis = normalize(vec2(cov_xy, lambda_major - cov_xx));
            } else {
                major_axis = cov_xx >= cov_yy ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
            }
            vec2 minor_axis = vec2(-major_axis.y, major_axis.x);
            vec2 gaussian_offset =
                major_axis * in_corner.x * sigma_major +
                minor_axis * in_corner.y * sigma_minor;
            pixel_offset = mix(pixel_offset, gaussian_offset, representation);
        }
        vec2 ndc = clip.xy / clip.w;
        ndc += pixel_offset * (2.0 / u_viewport);
        gl_Position = vec4(ndc * clip.w, clip.z, clip.w);
        v_opacity = mix(u_point_opacity, in_opacity * u_splat_opacity, representation);
    }
    v_local = in_corner;
    v_color = in_color;
    v_depth = clamp((depth - u_near) / max(u_far - u_near, 0.0001), 0.0, 1.0);
}
"""


_FRAGMENT_SHADER = r"""
#version 330

in vec2 v_local;
in vec3 v_color;
in float v_opacity;
uniform float u_exposure;

layout(location = 0) out vec4 out_color;

void main() {
    float power = -0.5 * dot(v_local, v_local);
    if (power < -8.0) {
        discard;
    }
    float alpha = min(0.999, exp(power) * v_opacity);
    if (alpha < (1.0 / 255.0)) {
        discard;
    }
    out_color = vec4(clamp(v_color * exp2(u_exposure), 0.0, 1.0), alpha);
}
"""


_COMPOSITE_VERTEX_SHADER = r"""
#version 330

out vec2 v_uv;

void main() {
    vec2 position = vec2(
        (gl_VertexID == 1) ? 3.0 : -1.0,
        (gl_VertexID == 2) ? 3.0 : -1.0
    );
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


_BACKGROUND_FRAGMENT_SHADER = r"""
#version 330

uniform vec3 u_background;
uniform float u_background_gradient;

in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 centered = v_uv - vec2(0.5);
    float glow = 1.0 - smoothstep(0.0, 0.72, length(centered));
    vec3 background = u_background * mix(
        1.0 - u_background_gradient,
        1.0 + u_background_gradient,
        glow
    );
    out_color = vec4(clamp(background, 0.0, 1.0), 1.0);
}
"""


_UNPREMULTIPLY_FRAGMENT_SHADER = r"""
#version 330

uniform sampler2D u_composite;

in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main() {
    vec4 composite = texture(u_composite, v_uv);
    vec3 straight_color = composite.a > (1.0 / 65535.0)
        ? composite.rgb / composite.a
        : vec3(0.0);
    out_color = vec4(clamp(straight_color, 0.0, 1.0), composite.a);
}
"""
