let recognizer;

function predictWord() {
    // Array of words that the recognizer is trained to recognize.
    const words = recognizer.wordLabels();
    recognizer.listen(({ scores }) => {
        // Turn scores into a list of (score,word) pairs.
        scores = Array.from(scores).map((s, i) => ({ score: s, word: words[i] }));
        // Find the most probable word.
        scores.sort((s1, s2) => s2.score - s1.score);
        document.querySelector('#console').textContent = scores[0].word;
    }, { probabilityThreshold: 0.85 });
}

async function buildModel() {
    //model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/model.json');
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/calvinlee1102/drug_webcam_test/master/mymodel.json');
    //const uploadJSONInput = document.getElementById('uploadvoicemodel');
    //const uploadWeightsInput = document.getElementById('uploadvoiceweight');
    //model = await tf.loadLayersModel(tf.io.browserFiles(
    //    [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
    listen();
    document.getElementById("status").innerHTML = "Jarvis Loaded";
}

async function loadjarvis() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    //predictWord();
    buildModel();
    //getVoiceModel();
}
//loadjarvis();

function collect(label) {
    if (recognizer.isListening()) {
        return recognizer.stopListening();
    }
    if (label == null) {
        return;
    }
    recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        examples.push({ vals, label });
        document.querySelector('#console').textContent =
            `${examples.length} examples collected`;
    }, {
            overlapFactor: 0.999,
            includeSpectrogram: true,
            invokeCallbackOnNoiseAndUnknown: true
        });
}

function normalize(x) {
    const mean = -100;
    const std = 10;
    return x.map(x => (x - mean) / std);
}

// One frame is ~23ms of audio.
const NUM_FRAMES = 43;
let examples = [];
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function voicetrain() {
    toggleButtons(false);
    const ys = tf.oneHot(examples.map(e => e.label), classes);
    const xsShape = [examples.length, ...INPUT_SHAPE];
    const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

    await model.fit(xs, ys, {
        batchSize: 25,
        epochs: 10,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.querySelector('#console').textContent =
                    `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
            }
        }
    });
    tf.dispose([xs, ys]);
    toggleButtons(true);
}

function toggleButtons(enable) {
    document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
    const size = tensors[0].length;
    const result = new Float32Array(tensors.length * size);
    tensors.forEach((arr, i) => result.set(arr, i * size));
    return result;
}

function getVoiceModel() {
    model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [NUM_FRAMES, 232, 1],
        kernelSize: [2, 8],
        filters: 8,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: [2, 4],
        filters: 32,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: [2, 4],
        filters: 32,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: [2, 4],
        filters: 32,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [1, 2]
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dropout({
        rate: 0.25
    }));
    model.add(tf.layers.dense({
        units: 2000,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    model.add(tf.layers.dropout({
        rate: 0.5
    }));
    model.add(tf.layers.dense({
        units: 6,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    const optimizer = tf.train.adam(0.01);
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
}

async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    if (labeltext[label] != 'noise' && labeltext[label] != "background") {
        commandcenter(labeltext[label]);
        //strshow = commandcenter(labeltext[label]);
        //document.getElementById('console').textContent = commandcenter(labeltext[label]);
    }
}

var str1 = "";
var str2 = "";
var str3 = "";
var command_status = "call_jarvis";
function commandcenter(labelstr) {
    str1 = str2;
    str2 = str3;
    str3 = labelstr;
    document.getElementById('console').innerHTML = str1 + "-" + str2 + "-" + str3;

    switch (command_status) {
        case "call_jarvis":
            if (str1 == str2 == str3 =="jarvis") { command_status = "wait_command"; }
            break;
        case "wait_command":
            if (str1 == str2 == str3 == "over") { command_status = "call_jarvis"; }
            if (str1 == str2 == str3 == "load_data") { command_status = "wait_confirm_load_data"; }
            if (str1 == str2 == str3 == "show_example") { command_status = "wait_confirm_show_example"; }
            if (str1 == str2 == str3 == "model_summary") { command_status = "wait_confirm_model_summary"; }
            if (str1 == str2 == str3 == "start_training") { command_status = "wait_confirm_start_training"; }
            break;
        case "wait_confirm_load_data":
            if (str1 == str2 == str3 == "confirm") { document.getElementById("load-data").click(); command_status = "data loading..."; }
            if (str1 == str2 == str3 == "negative") { command_status = "call_jarvis"; }
            break;
        case "wait_confirm_show_example":
            if (str1 == str2 == str3 == "confirm") { document.getElementById("show-examples").click(); command_status = "calculate example..."; }
            if (str1 == str2 == str3 == "negative") { command_status = "call_jarvis"; }
            break;
        case "wait_confirm_model_summary":
            if (str1 == str2 == str3 == "confirm") { modelinspection(); command_status = "collecting model data..."; }
            if (str1 == str2 == str3 == "negative") { command_status = "call_jarvis"; }
            break;
        case "wait_confirm_start_training":
            if (str1 == str2 == str3 == "confirm") { document.getElementById("start-training-1").click(); command_status = "training model..."; }
            if (str1 == str2 == str3 == "negative") { command_status = "call_jarvis"; }
            break;
        case "wait_confirm_matrix":
            if (str1 == str2 == str3 == "confirm") { document.getElementById("show-all").click(); command_status = "calculate confusion matrix..."; }
            if (str1 == str2 == str3 == "negative") { command_status = "call_jarvis"; }
            break;

    }
    document.getElementById('status').innerHTML = command_status;
}

function listen() {
    if (recognizer.isListening()) {
        recognizer.stopListening();
        return;
    }

    recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {
        const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
        const probs = model.predict(input);
        const predLabel = probs.argMax(1);
        await moveSlider(predLabel);
        tf.dispose([input, probs, predLabel]);
    }, {
            overlapFactor: 0.999,
            includeSpectrogram: true,
            invokeCallbackOnNoiseAndUnknown: true,
            probabilityThreshold: 0.9
        });
}
async function UploadModelFile() {
    const uploadJSONInput = document.getElementById('uploadmodel');
    const uploadWeightsInput = document.getElementById('uploadweight');
    model = await tf.loadLayersModel(tf.io.browserFiles(
        [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
}