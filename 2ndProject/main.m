% Based on the simulation, this script plots the vehicle movement at the same graph with the obstacle

% (X, Y) coordinates of the obstacle points
obstacle_X = [5, 5, 6, 6, 7, 7, 10];
obstacle_Y = [0, 1, 1, 2, 2, 3, 3];
figure;

% Plot of the movement environment
plot(obstacle_X, obstacle_Y, 'm', 'LineWidth', 1);

% Graph title, axis labels and limits
title('Vehicle Movement');
xlabel('X Position (m)');   
ylabel('Y Position (m)');
xlim([0, 12]);
ylim([0, 4]);
hold on;

% Desired point coordinates
target_X = 10;
target_Y = 3.2;

% Plot the desired point
plot(target_X, target_Y, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 5);

% Plot the (X,Y) vehicle route
plot(out.X.Data, out.Y.Data, 'g');
