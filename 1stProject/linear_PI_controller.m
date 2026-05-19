clc;
clear all;

% Βασικές Παράμετροι Ελεγκτή
Kp_val = 1.5;
zero_val = -0.35;
gain_adj = 1;
G_total = 25 * gain_adj * Kp_val;
Ki_val = Kp_val * (-zero_val);

% Συνάρτηση Μεταφοράς Ελεγκτή
controller_tf = zpk(zero_val, 0, Kp_val);
plant_tf = zpk([], [-0.1 -10], 25);

% Γεωμετρικός Τόπος Ριζών
open_loop_sys = controller_tf * plant_tf;
figure;
rlocus(open_loop_sys);

% Απόκριση στο Βήμα
closed_loop_sys = feedback(open_loop_sys, 1, -1);
figure;
step(closed_loop_sys);
