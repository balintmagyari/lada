/* What each graph page draws by default, and the conventions every graph follows.

   A preset names the x quantity (with fall-backs), the left-axis quantities, the choices
   offered for the right axis, and which kind of table it expects. The conventions are the
   ones rheologists read without a legend: one colour per sample, G′ filled and G″ open,
   tan δ as triangles on the right axis.

   window.ladaRheoPresets: { presets, palettes, sizes, markerFor(quantity), defaultSpec(key) } */

(function () {
  "use strict";

  var PRESETS = {
    frequency_sweep: {
      title: "Frequency sweep",
      testTypes: ["frequency_sweep"],
      x: ["angular_frequency", "frequency"],
      y: ["storage_modulus", "loss_modulus"],
      right: ["tan_delta", "complex_viscosity", "phase_angle"],
      defaultRight: "",
      logX: true,
      logY: true
    },
    amplitude_sweep: {
      title: "Amplitude sweep",
      testTypes: ["amplitude_sweep"],
      x: ["strain", "stress"],
      y: ["storage_modulus", "loss_modulus"],
      right: ["tan_delta", "phase_angle"],
      defaultRight: "",
      logX: true,
      logY: true
    },
    temperature_sweep: {
      title: "Temperature sweep",
      testTypes: ["temperature_sweep"],
      x: ["temperature"],
      y: ["storage_modulus", "loss_modulus"],
      right: ["tan_delta", "complex_viscosity", "phase_angle"],
      defaultRight: "tan_delta",
      logX: false,
      logY: true
    },
    time_sweep: {
      title: "Time sweep",
      testTypes: ["time_sweep"],
      x: ["time"],
      y: ["storage_modulus", "loss_modulus"],
      right: ["tan_delta", "complex_viscosity", "phase_angle"],
      defaultRight: "tan_delta",
      logX: false,
      logY: true
    },
    flow_curve: {
      title: "Flow curve",
      testTypes: ["flow_curve"],
      x: ["shear_rate", "stress"],
      y: ["viscosity"],
      right: ["stress"],
      defaultRight: "",
      logX: true,
      logY: true
    },
    creep: {
      title: "Creep",
      testTypes: ["creep"],
      x: ["time"],
      y: ["compliance"],
      right: [],
      defaultRight: "",
      logX: true,
      logY: true
    },
    stress_relaxation: {
      title: "Stress relaxation",
      testTypes: ["stress_relaxation"],
      x: ["time"],
      y: ["relaxation_modulus"],
      right: [],
      defaultRight: "",
      logX: true,
      logY: true
    },
    master_curve: {
      title: "Master curve",
      testTypes: ["frequency_sweep"],
      x: ["angular_frequency", "frequency"],
      y: ["storage_modulus", "loss_modulus"],
      right: ["tan_delta", "complex_viscosity", "phase_angle"],
      defaultRight: "",
      logX: true,
      logY: true,
      // The sweeps are drawn after time-temperature superposition (see studio.js).
      superpose: true
    },
    custom: {
      title: "Custom plot",
      testTypes: [],
      x: [],
      y: [],
      right: [],
      defaultRight: "",
      logX: false,
      logY: false
    }
  };

  // Colours for samples. Each is mid-toned, so it reads on a white page and on a dark one.
  var PALETTES = {
    lada: {
      name: "Default",
      colors: ["#2f6fd0", "#d1495b", "#2f9a62", "#e08e1b", "#7a5cc9", "#1f9eaa", "#9a6b3c",
        "#5b6573"]
    },
    colorblind: {
      name: "Colour-blind safe",
      // Okabe and Ito, with grey in place of black so it shows on a dark page too.
      colors: ["#0072b2", "#d55e00", "#009e73", "#e69f00", "#cc79a7", "#56b4e9", "#999999",
        "#f0e442"]
    },
    greyscale: {
      name: "Greyscale",
      colors: ["#4a4a4a", "#7a7a7a", "#a3a3a3", "#c7c7c7"]
    }
  };

  // Marker per quantity: same quantity, same shape, whichever sample it belongs to.
  var MARKERS = {
    storage_modulus: { symbol: "circle", filled: true },
    loss_modulus: { symbol: "circle", filled: false },
    tan_delta: { symbol: "triangle-up", filled: true },
    phase_angle: { symbol: "triangle-down", filled: true },
    complex_viscosity: { symbol: "square", filled: false },
    complex_modulus: { symbol: "diamond", filled: true },
    viscosity: { symbol: "circle", filled: true },
    stress: { symbol: "diamond", filled: false },
    compliance: { symbol: "circle", filled: true },
    relaxation_modulus: { symbol: "circle", filled: true }
  };
  var CYCLE = ["circle", "square", "triangle-up", "diamond", "triangle-down", "cross"];

  // The marker shapes every renderer can draw alike: Plotly here, matplotlib and pgfplots in
  // the exported figure.
  var SHAPES = [
    ["circle", "Circle"],
    ["square", "Square"],
    ["diamond", "Diamond"],
    ["triangle-up", "Triangle, up"],
    ["triangle-down", "Triangle, down"],
    ["cross", "Cross"]
  ];

  function markerFor(quantity, index) {
    return MARKERS[quantity] || { symbol: CYCLE[(index || 0) % CYCLE.length], filled: true };
  }

  // Figure sizes, width x height in millimetres: the ones export will use.
  var SIZES = {
    single: { name: "Single column, 85 mm", width: 85, height: 65 },
    double: { name: "Double column, 175 mm", width: 175, height: 100 },
    slide: { name: "Slide, 16:9", width: 254, height: 142.9 },
    square: { name: "Square, 85 mm", width: 85, height: 85 },
    // Width and height typed in (style.widthMm, style.heightMm).
    custom: { name: "Custom size", width: 85, height: 65 }
  };

  function axis(log) {
    return { log: !!log, min: "", max: "", title: "" };
  }

  function defaultSpec(key) {
    var preset = PRESETS[key] || PRESETS.custom;
    return {
      version: 1,
      preset: key,
      blocks: {}, // blockId -> { on: bool, label: string }
      right: preset.defaultRight,
      custom: { x: "", left: [], right: [] },
      // A graph page's own choice of what goes on each axis, once the user changes it: until
      // then (on: false) the page's preset decides.
      override: { on: false, x: "", left: [], right: [] },
      // Per-sample colours (block id -> #rrggbb) and per-quantity marker styles (column
      // identifier -> { symbol, filled, size, color }); anything unset follows the defaults.
      styles: { samples: {}, series: {} },
      units: {}, // canonical unit -> display unit symbol
      // Master curve only: reference temperature (°C, "" for the middle sweep), which moduli
      // the overlap uses, the bT correction, and shifts set by hand (block id -> log aT).
      tts: { reference: "", use: "both", vertical: false, manual: {} },
      axes: { x: axis(preset.logX), y: axis(preset.logY), y2: axis(!!preset.defaultRight) },
      style: {
        palette: "lada",
        mode: "markers",
        markerSize: 7,
        lineWidth: 1.5,
        font: "sans",
        fontSize: 9,
        legend: "above",
        legendX: 0.02, // a dragged legend's top left corner, as fractions of the plot area
        legendY: 0.98,
        grid: false,
        size: "single",
        widthMm: 85,
        heightMm: 65
      }
    };
  }

  window.ladaRheoPresets = {
    presets: PRESETS,
    palettes: PALETTES,
    sizes: SIZES,
    markerFor: markerFor,
    shapes: SHAPES,
    defaultSpec: defaultSpec
  };
})();
