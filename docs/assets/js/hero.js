/* Conquer3D hero -- a raymarched implicit surface.
 *
 * Deliberately dependency-free WebGL1: the hero renders the same kind of object
 * the library consumes, a signed distance field, sampled per-pixel through a
 * fullscreen quad. A gyroid is blended into a sphere so the surface reads as a
 * genuine isosurface rather than decoration, and a faint lattice recalls the
 * voxel grid the extraction kernels walk.
 *
 * Degrades silently: if WebGL is unavailable the canvas is simply removed and
 * the CSS radial gradient behind it carries the section on its own.
 */
(function () {
  "use strict";

  var canvas = document.getElementById("hero-canvas");
  if (!canvas) return;

  var gl = canvas.getContext("webgl", { antialias: false, alpha: true, depth: false })
        || canvas.getContext("experimental-webgl", { antialias: false, alpha: true, depth: false });
  if (!gl) { canvas.remove(); return; }

  var VERT = [
    "attribute vec2 p;",
    "void main(){ gl_Position = vec4(p, 0.0, 1.0); }"
  ].join("\n");

  var FRAG = [
    "precision highp float;",
    "uniform vec2  uRes;",
    "uniform float uTime;",
    "uniform vec2  uMouse;",
    "uniform float uDark;",

    // --- signed distance field -------------------------------------------
    // Sphere carved by a gyroid: g(p) = sin x cos y + sin y cos z + sin z cos x
    "float sdf(vec3 p){",
    "  float t = uTime * 0.16;",
    "  vec3 q = p;",
    "  float c = cos(t), s = sin(t);",
    "  q.xz = mat2(c, -s, s, c) * q.xz;",
    "  q.xy = mat2(cos(t*0.6), -sin(t*0.6), sin(t*0.6), cos(t*0.6)) * q.xy;",
    "  float scale = 3.1 + 0.55 * sin(uTime * 0.29);",
    "  float gyroid = sin(q.x*scale)*cos(q.y*scale)",
    "               + sin(q.y*scale)*cos(q.z*scale)",
    "               + sin(q.z*scale)*cos(q.x*scale);",
    "  float shell = length(q) - 1.16;",
    "  float thick = 0.34 + 0.12 * sin(uTime * 0.21);",
    "  float g = (abs(gyroid) - thick) / scale;",
    "  return max(shell, g) * 0.62;",
    "}",

    "vec3 normalAt(vec3 p){",
    "  vec2 e = vec2(0.0022, 0.0);",
    "  return normalize(vec3(",
    "    sdf(p+e.xyy)-sdf(p-e.xyy),",
    "    sdf(p+e.yxy)-sdf(p-e.yxy),",
    "    sdf(p+e.yyx)-sdf(p-e.yyx)));",
    "}",

    // Accent ramp: CUDA green -> cyan -> violet
    "vec3 ramp(float t){",
    "  t = clamp(t, 0.0, 1.0);",
    "  vec3 a = vec3(0.463, 0.725, 0.0);",
    "  vec3 b = vec3(0.133, 0.827, 0.933);",
    "  vec3 c = vec3(0.655, 0.545, 0.980);",
    "  return t < 0.5 ? mix(a, b, t*2.0) : mix(b, c, (t-0.5)*2.0);",
    "}",

    "void main(){",
    "  vec2 uv = (gl_FragCoord.xy - 0.5*uRes) / uRes.y;",
    "  vec3 ro = vec3(0.0, 0.0, 3.25);",
    "  vec2 m  = uMouse * 0.26;",
    "  ro.xz = mat2(cos(m.x), -sin(m.x), sin(m.x), cos(m.x)) * ro.xz;",
    "  ro.y += m.y;",
    "  vec3 ta = vec3(0.0);",
    "  vec3 fw = normalize(ta - ro);",
    "  vec3 rt = normalize(cross(vec3(0.0,1.0,0.0), fw));",
    "  vec3 up = cross(fw, rt);",
    "  vec3 rd = normalize(uv.x*rt + uv.y*up + 1.45*fw);",

    "  float d = 0.0; float hit = 0.0; vec3 pos = ro;",
    "  for(int i = 0; i < 78; i++){",
    "    pos = ro + rd*d;",
    "    float s = sdf(pos);",
    "    if(s < 0.0016){ hit = 1.0; break; }",
    "    d += s;",
    "    if(d > 6.2) break;",
    "  }",

    "  vec3 col = vec3(0.0);",
    "  float alpha = 0.0;",
    "  if(hit > 0.5){",
    "    vec3 n = normalAt(pos);",
    "    vec3 l = normalize(vec3(0.62, 0.78, 0.5));",
    "    float diff = max(dot(n, l), 0.0);",
    "    float fres = pow(1.0 - max(dot(n, -rd), 0.0), 2.6);",
    "    float depth = clamp((d - 1.7) / 2.5, 0.0, 1.0);",
    "    col = ramp(0.5*depth + 0.42*fres + 0.16*diff);",
    "    col *= 0.34 + 0.82*diff;",
    "    col += fres * 0.5 * ramp(0.78);",
    "    alpha = (0.90 - 0.42*depth) * (0.34 + 0.66*uDark);",
    "  }",

    // Voxel lattice, faint, fading with distance -- echoes the grid the
    // extraction kernels traverse.
    "  vec3 lp = ro + rd * 3.05;",
    "  vec3 gridv = abs(fract(lp*2.6) - 0.5);",
    "  float line = min(min(gridv.x, gridv.y), gridv.z);",
    "  float lat = smoothstep(0.045, 0.0, line) * 0.052 * (1.0 - alpha);",
    "  col += ramp(0.34) * lat;",
    "  alpha += lat;",

    "  gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));",
    "}"
  ].join("\n");

  function compile(type, src) {
    var sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.warn("hero shader:", gl.getShaderInfoLog(sh));
      return null;
    }
    return sh;
  }

  var vs = compile(gl.VERTEX_SHADER, VERT);
  var fs = compile(gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) { canvas.remove(); return; }

  var prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { canvas.remove(); return; }
  gl.useProgram(prog);

  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
  var loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  var uRes   = gl.getUniformLocation(prog, "uRes");
  var uTime  = gl.getUniformLocation(prog, "uTime");
  var uMouse = gl.getUniformLocation(prog, "uMouse");
  var uDark  = gl.getUniformLocation(prog, "uDark");

  var mouse = { x: 0, y: 0 }, target = { x: 0, y: 0 };
  window.addEventListener("mousemove", function (e) {
    target.x = (e.clientX / window.innerWidth - 0.5) * 2;
    target.y = (e.clientY / window.innerHeight - 0.5) * 2;
  }, { passive: true });

  function resize() {
    // Cap DPR: the shader is fill-rate bound and the hero is decorative.
    var dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    var W = Math.floor(w * dpr), H = Math.floor(h * dpr);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W; canvas.height = H;
      gl.viewport(0, 0, W, H);
    }
  }
  window.addEventListener("resize", resize);

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var visible = true;
  document.addEventListener("visibilitychange", function () { visible = !document.hidden; });

  // Pause once the hero is fully scrolled past.
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting && !document.hidden;
    }, { threshold: 0 }).observe(canvas);
  }

  var start = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible) return;
    resize();
    mouse.x += (target.x - mouse.x) * 0.045;
    mouse.y += (target.y - mouse.y) * 0.045;
    var t = reduced ? 8.0 : (now - start) / 1000;
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.uniform1f(uTime, t);
    gl.uniform2f(uMouse, mouse.x, mouse.y);
    gl.uniform1f(uDark, document.documentElement.getAttribute("data-theme") === "light" ? 0.0 : 1.0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  resize();
  requestAnimationFrame(frame);
})();
