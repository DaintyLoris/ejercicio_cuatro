% Cargando el conjunto de datos Iris
load fisheriris;
X = meas(:, 3:4);  % Tomando las dos últimas características (longitud y ancho del pétalo)
y = zeros(size(species, 1), 1);  % Incializando las etiquetas como 0's

% Asignando etiquetas a las clases
y(strcmp(species, 'setosa')) = 1;
y(strcmp(species, 'versicolor')) = 2;
y(strcmp(species, 'virginica')) = 3;

% Creando un perceptrón multicapa
hiddenLayerSize = 10;  % Número de neuronas en la capa oculta (ajusta según sea necesario)
net = patternnet(hiddenLayerSize);

% Dividiendo los datos en entrenamiento (80%) y prueba (20%)
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.0;
net.divideParam.testRatio = 0.2;

% Entrenando el modelo
net.trainParam.epochs = 1000;  % Número de épocas de entrenamiento
net = train(net, X', y');

% Realizando predicciones
y_pred = net(X');

% Visualizando la separación de clases
figure;
gscatter(X(:, 1), X(:, 2), species, 'rgb', 'osd');
title('Separación de las Clases de Iris Setosa con Perceptrón Multicapa');
xlabel('Longitud del Pétalo');
ylabel('Ancho del Pétalo');

% Calculando la precisión en el conjunto de entrenamiento
accuracy = sum(round(y_pred) == y) / length(y);
disp(['Precisión en el conjunto de entrenamiento: ', num2str(accuracy * 100), '%']);
