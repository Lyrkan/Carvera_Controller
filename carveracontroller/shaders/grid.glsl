// XY reference grid on a single quad
// vs_coords are scene-space scaled mm (same as toolpath position before center_offset).
// The quad is world-anchored at the origin, only grid_size scales to cover the view.

---vertex
$HEADER$
attribute vec3 position;

uniform mat4 center_offset;
uniform mat4 view_mat;
uniform mat4 proj_mat;
uniform float grid_size;

varying vec2 vs_coords;

void main()
{
    vec2 scene_xy = position.xy * grid_size;

    vs_coords = scene_xy;
    tex_coord0 = vec2(0.0);
    gl_Position = proj_mat * view_mat * center_offset * vec4(scene_xy, -0.0005, 1.0);
}

---fragment
#ifdef GL_ES
    #ifdef GL_OES_standard_derivatives
        #extension GL_OES_standard_derivatives : enable
        #define GRID_HAS_FWIDTH 1
    #endif
#else
    #define GRID_HAS_FWIDTH 1
#endif

$HEADER$

varying vec2 vs_coords;

uniform float grid_visible;
uniform float subcell_size;
uniform float cell_size;
uniform vec3 color_minor;
uniform vec3 color_major;
uniform vec3 color_axis_x;
uniform vec3 color_axis_y;

float grid_mod(float x, float y)
{
    return x - y * floor(x / y);
}

void main()
{
    if (grid_visible < 0.5) {
        discard;
    }

    float half_subcell = subcell_size * 0.5;
    float half_cell = cell_size * 0.5;

    vec2 subcell_coords = vec2(
        grid_mod(vs_coords.x + half_subcell, subcell_size),
        grid_mod(vs_coords.y + half_subcell, subcell_size)
    );
    vec2 cell_coords = vec2(
        grid_mod(vs_coords.x + half_cell, cell_size),
        grid_mod(vs_coords.y + half_cell, cell_size)
    );

    vec2 dist_subcell = abs(subcell_coords - half_subcell);
    vec2 dist_cell = abs(cell_coords - half_cell);

    float line_thickness = subcell_size * 0.01;

#ifdef GRID_HAS_FWIDTH
    // Cap derivative-based AA so zoomed-out lines do not merge into solid black.
    vec2 d = min(fwidth(vs_coords), vec2(subcell_size * 0.12));
    vec2 adj = 0.5 * (line_thickness + d);
#else
    // Fallback for GLES2 devices without OES_standard_derivatives.
    vec2 adj = vec2(line_thickness * 0.5);
#endif

    float on_axis_y = 1.0 - step(adj.x, abs(vs_coords.x));
    float on_axis_x = 1.0 - step(adj.y, abs(vs_coords.y));

    // Major/minor lines are per-axis; suppress X-directed lines near Y axis (and vice versa)
    // so grid lines do not show as a grey halo around colored axes.
    float on_minor_x = (1.0 - step(adj.x, dist_subcell.x)) * (1.0 - on_axis_y);
    float on_minor_y = (1.0 - step(adj.y, dist_subcell.y)) * (1.0 - on_axis_x);
    float on_major_x = (1.0 - step(adj.x, dist_cell.x)) * (1.0 - on_axis_y);
    float on_major_y = (1.0 - step(adj.y, dist_cell.y)) * (1.0 - on_axis_x);

    vec3 color = vec3(0.0);
    color = mix(color, color_axis_y, on_axis_y);
    color = mix(color, color_axis_x, on_axis_x * (1.0 - on_axis_y));
    float show_major = max(on_major_x, on_major_y);
    color = mix(color, color_major, show_major);
    float show_minor = max(on_minor_x, on_minor_y) * (1.0 - show_major);
    color = mix(color, color_minor, show_minor);

    if (length(color) < 0.01) {
        discard;
    }

    gl_FragColor = vec4(color, 0.5) * texture2D(texture0, tex_coord0);
}
