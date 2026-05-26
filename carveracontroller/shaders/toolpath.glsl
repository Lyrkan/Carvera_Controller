// G-code toolpath line shader (line_strip with playback trim and type filter)

---vertex
$HEADER$
attribute vec3 position;
attribute vec3 color_att;
attribute float type;
attribute float vertex_id;
attribute float distance_id;
attribute float vertex_tool;
attribute float vertex_feed;

uniform mat4 center_offset;
uniform mat4 rotation_mat;
uniform mat4 view_mat;
uniform mat4 proj_mat;

varying vec3 vs_color;
varying float vs_vertex_id;
varying float vs_distance_id;
varying float vs_vertex_type;
varying float vs_vertex_feed;

void main()
{
    vs_color = color_att;
    vs_vertex_id = vertex_id;
    vs_vertex_type = vertex_tool;
    vs_vertex_feed = vertex_feed;
    vs_distance_id = distance_id;

    tex_coord0 = vec2(0.0);
    vec4 world_pos = vec4(position, 1.0);
    if (vertex_id < 0.0) {
        gl_Position = proj_mat * view_mat * center_offset * world_pos;
    } else {
        gl_Position = proj_mat * view_mat * rotation_mat * center_offset * world_pos;
    }
}

---fragment
$HEADER$

varying vec3 vs_color;
varying float vs_vertex_id;
varying float vs_distance_id;
varying float vs_vertex_type;
varying float vs_vertex_feed;

// Kivy uniforms are float-only; -1 means show the full toolpath
uniform float display_count;
// Decimal-encoded type filter from set_display_mask (e.g. 1=type1, 10=type2, 11=types1+2)
uniform float vertex_type_display;
// 0 = move type, 1 = tool, 2 = feed speed
uniform float color_scheme;
uniform float feed_min;
uniform float feed_max;

vec3 tool_palette_color(float idx)
{
    float i = mod(floor(idx + 0.5), 10.0);
    if (abs(i - 0.0) < 0.5) return vec3(0.406684, 0.735902, 0.235489);
    if (abs(i - 1.0) < 0.5) return vec3(0.000000, 0.459774, 0.840728);
    if (abs(i - 2.0) < 0.5) return vec3(0.779915, 0.319537, 0.130857);
    if (abs(i - 3.0) < 0.5) return vec3(0.740127, 0.236840, 0.700182);
    if (abs(i - 4.0) < 0.5) return vec3(0.000000, 0.755849, 0.602221);
    if (abs(i - 5.0) < 0.5) return vec3(0.825216, 0.043248, 0.043248);
    if (abs(i - 6.0) < 0.5) return vec3(0.894806, 0.717161, 0.000000);
    if (abs(i - 7.0) < 0.5) return vec3(0.128923, 0.578319, 0.877916);
    if (abs(i - 8.0) < 0.5) return vec3(0.431518, 0.268501, 0.839063);
    return vec3(0.248716, 0.777237, 0.402157);
}

vec3 speed_colormap(float t)
{
    t = clamp(t, 0.0, 1.0);
    if (t < 0.33) {
        return mix(vec3(0.2, 0.4, 0.9), vec3(0.1, 0.7, 0.5), t / 0.33);
    }
    if (t < 0.66) {
        return mix(vec3(0.1, 0.7, 0.5), vec3(0.95, 0.85, 0.15), (t - 0.33) / 0.33);
    }
    return mix(vec3(0.95, 0.85, 0.15), vec3(0.9, 0.25, 0.2), (t - 0.66) / 0.34);
}

float mask_digit(float mask, float place)
{
    return mod(floor(mask / place), 10.0);
}

bool is_vertex_type_enabled(float vertex_type, float type_mask)
{
    float mask = floor(type_mask + 0.1);
    float vtype = floor(vertex_type + 0.1);
    if (mask < 0.5) {
        return true;
    }
    if (vtype < 0.5) {
        return false;
    }
    if (abs(vtype - 1.0) < 0.5 && mask_digit(mask, 1.0) >= 1.0) return true;
    if (abs(vtype - 2.0) < 0.5 && mask_digit(mask, 10.0) >= 1.0) return true;
    if (abs(vtype - 3.0) < 0.5 && mask_digit(mask, 100.0) >= 1.0) return true;
    if (abs(vtype - 4.0) < 0.5 && mask_digit(mask, 1000.0) >= 1.0) return true;
    if (abs(vtype - 5.0) < 0.5 && mask_digit(mask, 10000.0) >= 1.0) return true;
    if (abs(vtype - 6.0) < 0.5 && mask_digit(mask, 100000.0) >= 1.0) return true;
    if (abs(vtype - 7.0) < 0.5 && mask_digit(mask, 1000000.0) >= 1.0) return true;
    return false;
}

void main()
{
    if (display_count > -1.0 && vs_distance_id > display_count) {
        discard;
    }

    if (!is_vertex_type_enabled(vs_vertex_type, vertex_type_display)) {
        discard;
    }

    vec3 color;
    if (color_scheme < 0.5) {
        color = vs_color;
        // Rapid moves carry red in the vertex color; normalize to solid red
        if (color.r > 0.0) {
            color = vec3(1.0, 0.0, 0.0);
        }
    } else if (color_scheme < 1.5) {
        float tool_num = floor(vs_vertex_type + 0.5);
        float palette_idx = mod(max(tool_num, 1.0) - 1.0, 10.0);
        color = tool_palette_color(palette_idx);
    } else {
        if (vs_vertex_feed < 0.5) {
            color = vec3(1.0, 0.0, 0.0);
        } else {
            float span = max(feed_max - feed_min, 1.0);
            float t = clamp((vs_vertex_feed - feed_min) / span, 0.0, 1.0);
            color = speed_colormap(t);
        }
    }
    gl_FragColor = vec4(color, 1.0) * texture2D(texture0, tex_coord0);
}
