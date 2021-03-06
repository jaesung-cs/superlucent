#ifndef VKPBD_FLUID_COLOR_GLSL_
#define VKPBD_FLUID_COLOR_GLSL_

// Util functions generating colors
vec3 BlueRedColor(float normalized_intensity) // in [0, 1]
{
  const vec3 blue_red_colormap[] = 
  {
    uvec3(25,  82,  255),
    uvec3(25,  82,  255),
    uvec3(25,  82,  255),
    uvec3(27,  84,  255),
    uvec3(28,  85,  255),
    uvec3(30,  86,  255),
    uvec3(31,  87,  255),
    uvec3(33,  88,  255),
    uvec3(34,  89,  255),
    uvec3(36,  90,  255),
    uvec3(37,  92,  255),
    uvec3(39,  93,  255),
    uvec3(40,  94,  255),
    uvec3(42,  95,  255),
    uvec3(43,  96,  255),
    uvec3(45,  97,  255),
    uvec3(46,  98,  255),
    uvec3(48,  100, 255),
    uvec3(49,  101, 255),
    uvec3(51,  102, 255),
    uvec3(53,  103, 255),
    uvec3(54,  104, 255),
    uvec3(56,  105, 255),
    uvec3(57,  106, 255),
    uvec3(59,  108, 255),
    uvec3(60,  109, 255),
    uvec3(62,  110, 255),
    uvec3(63,  111, 255),
    uvec3(65,  112, 255),
    uvec3(66,  113, 255),
    uvec3(68,  115, 255),
    uvec3(69,  116, 255),
    uvec3(71,  117, 255),
    uvec3(72,  118, 255),
    uvec3(74,  119, 255),
    uvec3(75,  120, 255),
    uvec3(77,  121, 255),
    uvec3(79,  123, 255),
    uvec3(80,  124, 255),
    uvec3(82,  125, 255),
    uvec3(83,  126, 255),
    uvec3(85,  127, 255),
    uvec3(86,  128, 255),
    uvec3(88,  129, 255),
    uvec3(89,  131, 255),
    uvec3(91,  132, 255),
    uvec3(92,  133, 255),
    uvec3(94,  134, 255),
    uvec3(95,  135, 255),
    uvec3(97,  136, 255),
    uvec3(98,  137, 255),
    uvec3(100, 139, 255),
    uvec3(102, 140, 255),
    uvec3(103, 141, 255),
    uvec3(105, 142, 255),
    uvec3(106, 143, 255),
    uvec3(108, 144, 255),
    uvec3(109, 145, 255),
    uvec3(111, 147, 255),
    uvec3(112, 148, 255),
    uvec3(114, 149, 255),
    uvec3(115, 150, 255),
    uvec3(117, 151, 255),
    uvec3(118, 152, 255),
    uvec3(120, 154, 255),
    uvec3(121, 155, 255),
    uvec3(123, 156, 255),
    uvec3(124, 157, 255),
    uvec3(126, 158, 255),
    uvec3(128, 159, 255),
    uvec3(129, 160, 255),
    uvec3(131, 162, 255),
    uvec3(132, 163, 255),
    uvec3(134, 164, 255),
    uvec3(135, 165, 255),
    uvec3(137, 166, 255),
    uvec3(138, 167, 255),
    uvec3(140, 168, 255),
    uvec3(141, 170, 255),
    uvec3(143, 171, 255),
    uvec3(144, 172, 255),
    uvec3(146, 173, 255),
    uvec3(147, 174, 255),
    uvec3(149, 175, 255),
    uvec3(150, 176, 255),
    uvec3(152, 178, 255),
    uvec3(154, 179, 255),
    uvec3(155, 180, 255),
    uvec3(157, 181, 255),
    uvec3(158, 182, 255),
    uvec3(160, 183, 255),
    uvec3(161, 185, 255),
    uvec3(163, 186, 255),
    uvec3(164, 187, 255),
    uvec3(166, 188, 255),
    uvec3(167, 189, 255),
    uvec3(169, 190, 255),
    uvec3(170, 191, 255),
    uvec3(172, 193, 255),
    uvec3(173, 194, 255),
    uvec3(175, 195, 255),
    uvec3(176, 196, 255),
    uvec3(178, 197, 255),
    uvec3(180, 198, 255),
    uvec3(181, 199, 255),
    uvec3(183, 201, 255),
    uvec3(184, 202, 255),
    uvec3(186, 203, 255),
    uvec3(187, 204, 255),
    uvec3(189, 205, 255),
    uvec3(190, 206, 255),
    uvec3(192, 207, 255),
    uvec3(193, 209, 255),
    uvec3(195, 210, 255),
    uvec3(196, 211, 255),
    uvec3(198, 212, 255),
    uvec3(199, 213, 255),
    uvec3(201, 214, 255),
    uvec3(202, 215, 255),
    uvec3(204, 217, 255),
    uvec3(206, 218, 255),
    uvec3(207, 219, 255),
    uvec3(209, 220, 255),
    uvec3(210, 221, 255),
    uvec3(212, 222, 255),
    uvec3(213, 224, 255),
    uvec3(215, 225, 255),
    uvec3(216, 226, 255),
    uvec3(255, 216, 216),
    uvec3(255, 215, 215),
    uvec3(255, 213, 213),
    uvec3(255, 212, 212),
    uvec3(255, 210, 210),
    uvec3(255, 209, 209),
    uvec3(255, 207, 207),
    uvec3(255, 206, 206),
    uvec3(255, 204, 204),
    uvec3(255, 202, 202),
    uvec3(255, 201, 201),
    uvec3(255, 199, 199),
    uvec3(255, 198, 198),
    uvec3(255, 196, 196),
    uvec3(255, 195, 195),
    uvec3(255, 193, 193),
    uvec3(255, 192, 192),
    uvec3(255, 190, 190),
    uvec3(255, 189, 189),
    uvec3(255, 187, 187),
    uvec3(255, 186, 186),
    uvec3(255, 184, 184),
    uvec3(255, 183, 183),
    uvec3(255, 181, 181),
    uvec3(255, 180, 180),
    uvec3(255, 178, 178),
    uvec3(255, 176, 176),
    uvec3(255, 175, 175),
    uvec3(255, 173, 173),
    uvec3(255, 172, 172),
    uvec3(255, 170, 170),
    uvec3(255, 169, 169),
    uvec3(255, 167, 167),
    uvec3(255, 166, 166),
    uvec3(255, 164, 164),
    uvec3(255, 163, 163),
    uvec3(255, 161, 161),
    uvec3(255, 160, 160),
    uvec3(255, 158, 158),
    uvec3(255, 157, 157),
    uvec3(255, 155, 155),
    uvec3(255, 154, 154),
    uvec3(255, 152, 152),
    uvec3(255, 150, 150),
    uvec3(255, 149, 149),
    uvec3(255, 147, 147),
    uvec3(255, 146, 146),
    uvec3(255, 144, 144),
    uvec3(255, 143, 143),
    uvec3(255, 141, 141),
    uvec3(255, 140, 140),
    uvec3(255, 138, 138),
    uvec3(255, 137, 137),
    uvec3(255, 135, 135),
    uvec3(255, 134, 134),
    uvec3(255, 132, 132),
    uvec3(255, 131, 131),
    uvec3(255, 129, 129),
    uvec3(255, 128, 128),
    uvec3(255, 126, 126),
    uvec3(255, 124, 124),
    uvec3(255, 123, 123),
    uvec3(255, 121, 121),
    uvec3(255, 120, 120),
    uvec3(255, 118, 118),
    uvec3(255, 117, 117),
    uvec3(255, 115, 115),
    uvec3(255, 114, 114),
    uvec3(255, 112, 112),
    uvec3(255, 111, 111),
    uvec3(255, 109, 109),
    uvec3(255, 108, 108),
    uvec3(255, 106, 106),
    uvec3(255, 105, 105),
    uvec3(255, 103, 103),
    uvec3(255, 101, 101),
    uvec3(255, 100, 100),
    uvec3(255, 98,  98),
    uvec3(255, 97,  97),
    uvec3(255, 95,  95),
    uvec3(255, 94,  94),
    uvec3(255, 92,  92),
    uvec3(255, 91,  91),
    uvec3(255, 89,  89),
    uvec3(255, 88,  88),
    uvec3(255, 86,  86),
    uvec3(255, 85,  85),
    uvec3(255, 83,  83),
    uvec3(255, 82,  82),
    uvec3(255, 80,  80),
    uvec3(255, 79,  79),
    uvec3(255, 77,  77),
    uvec3(255, 75,  75),
    uvec3(255, 74,  74),
    uvec3(255, 72,  72),
    uvec3(255, 71,  71),
    uvec3(255, 69,  69),
    uvec3(255, 68,  68),
    uvec3(255, 66,  66),
    uvec3(255, 65,  65),
    uvec3(255, 63,  63),
    uvec3(255, 62,  62),
    uvec3(255, 60,  60),
    uvec3(255, 59,  59),
    uvec3(255, 57,  57),
    uvec3(255, 56,  56),
    uvec3(255, 54,  54),
    uvec3(255, 53,  53),
    uvec3(255, 51,  51),
    uvec3(255, 49,  49),
    uvec3(255, 48,  48),
    uvec3(255, 46,  46),
    uvec3(255, 45,  45),
    uvec3(255, 43,  43),
    uvec3(255, 42,  42),
    uvec3(255, 40,  40),
    uvec3(255, 39,  39),
    uvec3(255, 37,  37),
    uvec3(255, 36,  36),
    uvec3(255, 34,  34),
    uvec3(255, 33,  33),
    uvec3(255, 31,  31),
    uvec3(255, 30,  30),
    uvec3(255, 28,  28),
    uvec3(255, 27,  27),
    uvec3(255, 25,  25),
    uvec3(255, 25,  25),
    uvec3(255, 25,  25),
  };

  int index = int(clamp(normalized_intensity, 0.f, 1.f) * 255.f);
  return blue_red_colormap[index] / 255.f;
}

#endif // VKPBD_FLUID_COLOR_GLSL_
