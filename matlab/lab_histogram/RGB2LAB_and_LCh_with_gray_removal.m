% RGB2LAB_and_LCh_with_gray_removal - Given an RGB input image (assumed to be 
%		in the range 0-1), identifies and removes pixels which appear to be
%		very close to gray, then returns the remaining pixel values converted 
%		to CIELAB and CIE L*C*h(a*b*) values.
function [L, a, b, C, h] = RGB2LAB_and_LCh_with_gray_removal(input_image);
	% -- Settings --
	THRESH = 0.94;	% Threshold for gray pixel rejection.
	% -- End Settings --

	% Extract RGB pixel values, convert to HSV.
	R = input_image(:,:,1); G = input_image(:,:,2); B = input_image(:,:,3);
	R = R(:); G = G(:); B = B(:);
	HSV = rgb2hsv([R.' G.' B.']);
	H = HSV(:,1); S = HSV(:,2); V = HSV(:,3);

	% Calculate criterion for identifying gray pixels, remove pixels which
	%	appear to be gray.
	K = (1-S).^3 + (1-V).^3;
	gray_pixels = find(K > THRESH^3);

	R(gray_pixels) = [];
	G(gray_pixels) = [];
	B(gray_pixels) = [];

	% Convert pixel values from sRGB to CIELAB, then to CIE L*C*h(a*b*).
	[L, a, b] = RGB2Lab(R, G, B);
	h = atan2(b, a);
	C = hypot(b, a);
