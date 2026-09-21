const RAMP_A = [0, 60, 140, 200, 255];

const RGB = {
  viridis:  [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
  inferno:  [[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,255,164]],
  magma:    [[0,0,4],[81,18,124],[183,55,121],[252,137,97],[252,253,191]],
  grayscale:[[0,0,0],[64,64,64],[128,128,128],[192,192,192],[255,255,255]],
};

export const COLORMAP_NAMES = ['viridis', 'inferno', 'magma', 'grayscale'];

export function colormapStops(name) {
  const rgb = RGB[name] || RGB.viridis;
  return rgb.map((c, i) => ({
    t: i / (rgb.length - 1),
    r: c[0], g: c[1], b: c[2], a: RAMP_A[i],
  }));
}
