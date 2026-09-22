// Heightmap / rotary stock. The mesh is a fixed grid; texture2 stores the field.

---vertex
$HEADER$

attribute vec3 v_pos;
attribute vec3 v_normal;
attribute vec2 v_uv;
attribute float v_role;
attribute float v_extra;

uniform mat4 center_offset;
uniform mat4 view_mat;
uniform mat4 proj_mat;
uniform mat4 rotation_mat;
uniform float vertex_scale;
uniform sampler2D texture2;
uniform float field_mode;
uniform vec2 field_size;
uniform float field_lo;
uniform float field_span;
uniform float field_floor;
uniform float field_cell;
uniform vec2 field_axis_yz;

varying vec3 normal_vec;
varying float world_z;
varying vec3 stock_pos;
varying vec3 stock_n;
varying vec3 view_pos;
varying vec2 field_uv;
varying float field_role;

// xy = (value, 1) for a solid or through-cut texel, (0, 0) when outside or off the grid.
vec2 field_tap(vec2 uv)
{
    vec2 st = uv;
    if (field_mode > 1.5) {
        st.y = st.y - floor(st.y);
    }
    vec2 cell = floor(st * field_size);
    if (cell.x < 0.0 || cell.y < 0.0 || cell.x >= field_size.x || cell.y >= field_size.y) {
        return vec2(0.0, 0.0);
    }
    vec4 tex = texture2D(texture2, (cell + vec2(0.5, 0.5)) / field_size);
    float flag = tex.b * 255.0;
    if (flag < 0.5) {
        return vec2(0.0, 0.0);
    }
    // 16-bit fixed point in R/G. tex.r is hi/255, so the high byte weighs 255*256.
    // The constant stays below 1 and mediump never sees a 65280 intermediate.
    float unit = tex.r * (65280.0 / 65535.0) + tex.g * (255.0 / 65535.0);
    float value = field_lo + unit * field_span;
    if (field_mode > 1.5) {
        value = max(value, field_floor);
    }
    return vec2(value, 1.0);
}

// x = corner value, y = d(value)/dx, z = d(value)/dy in mm, w = sample count.
vec4 field_corner(vec2 uv)
{
    vec2 texel = vec2(1.0) / max(field_size, vec2(1.0));
    vec2 sw = field_tap(uv + vec2(-0.5, -0.5) * texel);
    vec2 se = field_tap(uv + vec2(0.5, -0.5) * texel);
    vec2 nw = field_tap(uv + vec2(-0.5, 0.5) * texel);
    vec2 ne = field_tap(uv + vec2(0.5, 0.5) * texel);
    float count = sw.y + se.y + nw.y + ne.y;
    float h = 0.0;
    if (count > 0.5) {
        h = (sw.x + se.x + nw.x + ne.x) / count;
    }
    float west_n = sw.y + nw.y;
    float east_n = se.y + ne.y;
    float south_n = sw.y + se.y;
    float north_n = nw.y + ne.y;
    float west = h;
    float east = h;
    float south = h;
    float north = h;
    if (west_n > 0.5) {
        west = (sw.x + nw.x) / west_n;
    }
    if (east_n > 0.5) {
        east = (se.x + ne.x) / east_n;
    }
    if (south_n > 0.5) {
        south = (sw.x + se.x) / south_n;
    }
    if (north_n > 0.5) {
        north = (nw.x + ne.x) / north_n;
    }
    float cell = max(field_cell, 1e-6);
    return vec4(h, (east - west) / cell, (north - south) / cell, count);
}

void main()
{
    vec4 corner = vec4(0.0);
    vec2 tap = vec2(0.0);
    // Tops and the top edge of a wall read one cell. Averaging the four
    // neighbors turned every step into a ramp a whole cell wide.
    if (v_role < 0.5 || (v_role > 1.5 && v_role < 2.5 && v_extra > 0.5)) {
        tap = field_tap(v_uv);
    } else if (v_role > 2.5 && !(v_role > 3.5 && v_extra < 0.5)) {
        corner = field_corner(v_uv);
    }
    float raised = v_pos.z;
    if (tap.y > 0.5) {
        raised = tap.x;
    } else if (corner.w > 0.5) {
        raised = corner.x;
    }
    vec3 displaced = v_pos;
    vec3 nrm = v_normal;

    if (v_role < 0.5) {
        displaced = vec3(v_pos.xy, raised);
    } else if (v_role < 1.5) {
        displaced = v_pos;
    } else if (v_role < 2.5) {
        float z = v_pos.z;
        if (v_extra > 0.5) {
            z = raised;
        }
        displaced = vec3(v_pos.xy, z);
    } else if (v_role < 3.5) {
        float r = 0.0;
        if (corner.w > 0.5) {
            r = corner.x;
        }
        float s = v_pos.y;
        float c = v_pos.z;
        float dtheta = 6.28318530718 / max(field_size.y, 1.0);
        float rx = corner.y;
        float rt = corner.z * max(field_cell, 1e-6) / dtheta;
        displaced = vec3(v_pos.x, field_axis_yz.x + r * s, field_axis_yz.y + r * c);
        nrm = vec3(-r * rx, r * s - rt * c, r * c + rt * s);
        if (dot(nrm, nrm) < 1e-8) {
            nrm = vec3(0.0, s, c);
        }
    } else if (v_extra < 0.5) {
        displaced = vec3(v_pos.x, field_axis_yz.x, field_axis_yz.y);
    } else {
        float r = 0.0;
        if (corner.w > 0.5) {
            r = corner.x;
        }
        displaced = vec3(v_pos.x, field_axis_yz.x + r * v_pos.y, field_axis_yz.y + r * v_pos.z);
    }

    vec3 scaled_pos = displaced * vertex_scale;
    stock_pos = scaled_pos;
    stock_n = nrm;
    vec4 rotated = rotation_mat * vec4(scaled_pos, 1.0);
    vec4 world = center_offset * rotated;
    vec4 eye_pos = view_mat * world;
    view_pos = eye_pos.xyz;
    normal_vec = (view_mat * rotation_mat * vec4(nrm, 0.0)).xyz;
    world_z = rotated.z;
    field_uv = v_uv;
    field_role = v_role;
    tex_coord0 = vec2(0.0);
    gl_Position = proj_mat * eye_pos;
}

---fragment
#ifdef GL_ES
    #ifdef GL_OES_standard_derivatives
        #extension GL_OES_standard_derivatives : enable
        #define FIELD_HAS_DERIVATIVES 1
    #endif
#else
    #define FIELD_HAS_DERIVATIVES 1
#endif

$HEADER$

varying vec3 normal_vec;
varying float world_z;
varying vec3 stock_pos;
varying vec3 stock_n;
varying vec3 view_pos;
varying vec2 field_uv;
varying float field_role;

uniform float stock_z_min;
uniform float stock_z_max;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float laser_enabled;
uniform float laser_mode;
uniform vec2 stock_xy_min;
uniform vec2 stock_xy_span;
uniform vec2 axis_yz;
uniform vec3 surface_color;
uniform vec3 interior_color;
uniform float use_two_tone;
uniform float use_height_tint;
uniform float surface_z_eps;
uniform float cylindrical_skin;
uniform float stock_radius;
uniform float metallic;
uniform float interior_metallic;
uniform float roughness;
uniform float interior_roughness;
uniform vec2 field_size;
uniform float field_mode;

void main()
{
    vec2 cell = floor(field_uv * field_size);
    cell = clamp(cell, vec2(0.0), field_size - vec2(1.0));
    float flag = texture2D(texture2, (cell + vec2(0.5)) / field_size).b * 255.0;
    bool open_cell = flag < 0.5 || flag > 1.5;
#ifdef FIELD_HAS_DERIVATIVES
    // Face normal, so a step is one flat plane instead of a smooth lighting smear.
    vec3 face_n = cross(dFdx(view_pos), dFdy(view_pos));
    float rise = dFdx(stock_pos.z) * dFdx(stock_pos.z) + dFdy(stock_pos.z) * dFdy(stock_pos.z);
    float run = dot(dFdx(stock_pos.xy), dFdx(stock_pos.xy)) + dot(dFdy(stock_pos.xy), dFdy(stock_pos.xy));
    bool steep = rise > run * 4.0;
#else
    vec3 face_n = vec3(0.0);
    bool steep = false;
#endif
    // Tops and bottoms drop outside cells and through-cuts. A through-cut's
    // sides are walls, so they stay all the way down to the bottom.
    // Shell and caps drop empty samples so a gap opens in the bar.
    if (field_role < 0.5) {
        if (open_cell && !steep) {
            discard;
        }
    } else if (field_role < 1.5) {
        if (open_cell) {
            discard;
        }
    } else if (field_role > 2.5) {
        if (flag < 0.5) {
            discard;
        }
    }

    vec3 n = normalize(normal_vec);
#ifdef FIELD_HAS_DERIVATIVES
    if (field_mode < 1.5 && dot(face_n, face_n) > 1e-12) {
        n = normalize(face_n);
    }
#endif
    vec3 light = normalize(vec3(0.35, 0.55, 1.0));
    float ndl = abs(dot(n, light));

    float is_surface = 1.0;
    vec3 base = surface_color;
    if (use_two_tone > 0.5) {
        float nlen = max(length(stock_n), 1e-6);
        if (cylindrical_skin > 0.5) {
            vec2 d = stock_pos.yz - axis_yz;
            float r = length(d);
            float cap = step(0.35, abs(stock_n.x) / nlen);
            float outward = step(0.0, dot(stock_n.yz, d));
            float at_od = step(stock_radius - surface_z_eps, r);
            is_surface = (1.0 - cap) * outward * at_od;
        } else {
            float up_face = step(0.35, stock_n.z / nlen);
            float at_top = step(stock_z_max - surface_z_eps, stock_pos.z);
            is_surface = up_face * at_top;
        }
        base = mix(interior_color, surface_color, is_surface);
    }

    vec3 albedo = base;
    if (use_height_tint > 0.5) {
        float z_span = max(stock_z_max - stock_z_min, 1e-6);
        float t = clamp((world_z - stock_z_min) / z_span, 0.0, 1.0);
        vec3 low = base * vec3(0.52, 0.48, 0.42);
        vec3 high = base * vec3(1.08, 1.05, 0.98);
        albedo = mix(low, high, t);
    }

    float metal = clamp(mix(interior_metallic, metallic, is_surface), 0.0, 1.0);
    float rough = clamp(mix(interior_roughness, roughness, is_surface), 0.04, 1.0);
    vec3 f0 = mix(vec3(0.04), albedo, metal);
    float wrap = 0.18 + 0.82 * ndl;
    vec3 diffuse = albedo * (1.0 - metal) * wrap;

    vec3 view_dir = normalize(-view_pos);
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }
    vec3 half_vec = normalize(light + view_dir);
    float ndh = max(dot(n, half_vec), 0.0);
    float ndv = max(dot(n, view_dir), 0.0);

    float spec_exp = min(exp2(10.0 * (1.0 - rough)), 256.0);
    float spec = pow(ndh, spec_exp);
    float fresnel_w = pow(clamp(1.0 - ndv, 0.0, 1.0), 5.0);
    vec3 fresnel = f0 + (vec3(1.0) - f0) * fresnel_w;
    vec3 specular = fresnel * spec;

    vec3 metal_fill = f0 * (0.32 + 0.68 * ndl) + f0 * fresnel_w * 0.45;

    vec3 color = diffuse + mix(vec3(0.0), metal_fill, metal) + specular;
    if (laser_enabled > 0.5) {
        vec2 uv;
        float facing;
        if (laser_mode < 0.5) {
            uv = (stock_pos.xy - stock_xy_min) / max(stock_xy_span, vec2(1e-6));
            facing = step(0.35, stock_n.z / max(length(stock_n), 1e-6));
        } else {
            vec2 d = stock_pos.yz - axis_yz;
            float theta = atan(d.x, d.y);
            uv.x = (stock_pos.x - stock_xy_min.x) / max(stock_xy_span.x, 1e-6);
            uv.y = theta * 0.15915494309;
            if (uv.y < 0.0) {
                uv.y += 1.0;
            }
            facing = step(0.0, dot(stock_n.yz, d));
        }
        float burn = texture2D(texture1, uv).r * facing;
        color = mix(color, color * vec3(0.22, 0.20, 0.18), burn);
    }
    gl_FragColor = vec4(color, 1.0) * texture2D(texture0, tex_coord0);
}
