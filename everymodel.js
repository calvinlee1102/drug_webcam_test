function getModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [imgW, imgH, channel],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    //var optimizer = tf.train.sgd(LEARNING_RATE);
    //alert(optimizertype);
    //if (optimizertype =='sgd') {
    //    optimizer = tf.train.sgd(LEARNING_RATE);
    //} else if (optimizertype == 'momentum') {
    //    optimizer = tf.train.momentum(LEARNING_RATE);
    //}
    //model.compile({
    //    optimizer: optimizer,
    //    loss: 'categoricalCrossentropy',
    //    metrics: ['accuracy']
    //});
    return model;
}