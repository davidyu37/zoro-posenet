const videoWidth = window.screen.width;
const videoHeight = window.screen.height;

let video;
let poseNet;
let poses = [];
let state = "normal";

let bandana,
  face,
  kitetsu,
  shusui,
  wado,
  sounds = [];
const custom_model = {
  model: "assets/models/model.json",
  metadata: "assets/models/model_meta.json",
  weights: "assets/models/model.weights.bin"
};

function preload() {
  bandana = loadImage("assets/bandana.png");
  face = loadImage("assets/zoro-face.png");
  kitetsu = loadImage("assets/kitetsu.png");
  shusui = loadImage("assets/shusui.png");
  wado = loadImage("assets/wado.png");

  const oni1 = loadSound("assets/oni-1.mp4");
  const giri1 = loadSound("assets/giri-1.mp4");

  sounds = [oni1, giri1];
}

function vidLoad() {
  console.log("video loaded");
  console.log(video.width, video.height);
  video.play();
}

function setup() {
  angleMode(DEGREES);
  createCanvas(videoWidth, videoHeight);
  video = createCapture(VIDEO);
  video.size(width, height);

  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(
    video,
    // {
    //   imageScaleFactor: 0.3,
    //   outputStride: 16,
    //   flipHorizontal: false,
    //   minConfidence: 0.8,
    //   maxPoseDetections: 5,
    //   scoreThreshold: 0.5,
    //   nmsRadius: 20,
    //   detectionType: "single",
    //   multiplier: 1.0
    // },
    modelReady
  );

  loadBrain();

  poseNet.on("pose", function(results) {
    poses = results;
    gotPoses(poses);
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  console.log("model loaded");
}

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (pose) {
      classifyPose(pose);
    }
  }
}

function loadBrain() {
  let options = {
    inputs: 34,
    outputs: 3,
    task: "classification",
    debug: true
  };
  brain = ml5.neuralNetwork(options);

  brain.load(custom_model, brainLoaded);
}

function brainLoaded() {
  console.log("pose classification ready!");
}

function classifyPose(pose) {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  }
}

function gotResult(error, results) {
  if (results[0] && results[1] && results[2]) {
    let normal, oni, giri;
    results.map(result => {
      if (result.label === "normal") {
        normal = result;
        return;
      }
      if (result.label === "oni") {
        oni = result;
        return;
      }
      if (result.label === "giri") {
        giri = result;
        return;
      }
    });
    if (oni.confidence > 0.9) {
      if (state === "normal") {
        console.log("switch to oni");
        sounds[0].play();
      }

      state = "oni";
      return;
    } else if (giri.confidence > 0.9) {
      if (state === "normal") {
        console.log("switch to giri");
        sounds[1].play();
      }
      state = "giri";
      return;
    } else {
      state = "normal";
    }
  }
}

function draw() {
  image(video, 0, 0, width, height);

  drawFace();
  checkPose();
}

let soundPlaying = false;

function checkPose() {
  if (poses[0]) {
    // fill(255, 0, 0);
    // textSize(200);
    // text(poses[0].pose.score, width / 8, height / 2);
    if (poses[0].pose.score > 0.8) {
      console.log(state);

      drawWado();
      drawKitetsu();
      drawShusui();

      // if (state === "oni") {
      //   drawWado();
      //   drawKitetsu();
      //   drawShusui();
      // }
      // if (state === "giri") {
      //   drawWado();
      //   drawKitetsu();
      //   drawShusui();
      // }
    }
  }
}

// Detect the position of head and add bandana
function drawBandana() {
  if (poses[0]) {
    const {
      pose: { leftEar, rightEar, nose, rightEye, leftEye }
    } = poses[0];

    if (leftEar && rightEar && nose && rightEye && leftEye) {
      const { x, y } = leftEar;
      const bandaWidth = (leftEar.x - rightEar.x) * 1.4;
      const bandanaHeight = bandaWidth * 0.75;
      const positionX = nose.x - bandaWidth * 0.4;
      const positionY = rightEye.y - bandanaHeight;
      image(bandana, positionX, positionY, bandaWidth, bandanaHeight);
    }
  }
}

function drawFace() {
  if (poses[0]) {
    const {
      pose: {
        leftEar,
        rightEar,
        nose,
        rightEye,
        leftEye,
        leftShoulder,
        rightShoulder
      }
    } = poses[0];

    const imageRatio = face.height / face.width;

    if (leftEar && rightEar && nose && rightEye && leftEye) {
      const { x, y } = leftEar;
      const faceWidth = (leftEar.x - rightEar.x) * 1.4;
      const faceHeight = faceWidth * imageRatio;
      const positionX = nose.x - faceWidth * 0.5;
      const positionY = rightEye.y - faceHeight / 2;
      push();

      const shoulderAnchor =
        rightShoulder.x + (leftShoulder.x - rightShoulder.x) / 2;
      const slope = (rightShoulder.y - nose.y) / (shoulderAnchor - nose.x);

      const headTilt = atan(slope);

      translate(shoulderAnchor, positionY + faceHeight);
      // if (headTilt > 0) {
      //   rotate(headTilt - 90);
      // } else {
      //   rotate(90 + headTilt);
      // }

      image(face, -faceWidth / 2, -faceHeight, faceWidth, faceHeight);
      pop();
    }
  }
}

function drawWado() {
  if (poses[0]) {
    const {
      pose: { leftEar, rightEar, rightEye, nose }
    } = poses[0];

    console.log(poses[0]);

    if (leftEar && rightEar && nose && rightEye) {
      const faceWidth = (leftEar.x - rightEar.x) * 1.4;
      const { x, y } = nose;
      const swordWidth = faceWidth * 3;
      const swordHeight = faceWidth / 4;
      const noseAndEyeX = nose.x - rightEye.x;
      const noseAndEyeY = nose.y - rightEye.y;
      const startX = x - swordWidth + noseAndEyeX;
      const startY = y + noseAndEyeY * 0.2;
      image(wado, startX, startY, swordWidth, swordHeight);
    }
  }
}

function drawKitetsu() {
  if (poses[0]) {
    const {
      pose: { leftWrist, leftShoulder, rightShoulder, leftElbow }
    } = poses[0];

    if (leftWrist && leftShoulder && rightShoulder && leftElbow) {
      const { x, y } = leftWrist;
      const bodyWidth = leftShoulder.x - rightShoulder.x;
      const swordHeight = bodyWidth * 2;
      const swordWidth = bodyWidth / 2;
      const dBetweenWristAndElbow = leftWrist.x - leftElbow.x;
      const dBetweenWristAndElbowY = leftWrist.y - leftElbow.y;
      const startX = x;
      const startY = y;
      push();

      const slope = dBetweenWristAndElbowY / dBetweenWristAndElbow;

      const swordTilt = atan(slope);

      translate(startX, startY);

      if (dBetweenWristAndElbow < 0) {
        scale(-1.0, 1.0);
        translate(swordWidth, 0);
      }

      // if (dBetweenWristAndElbowY < 0) {
      //   scale(1.0, -1.0);
      // }

      // rotate(swordTilt);
      image(
        kitetsu,
        dBetweenWristAndElbow * 0.3,
        -swordHeight + dBetweenWristAndElbowY,
        swordWidth,
        swordHeight
      );
      pop();
    }
  }
}
function drawShusui() {
  if (poses[0]) {
    const {
      pose: { rightWrist, leftShoulder, rightShoulder, rightElbow, leftEar }
    } = poses[0];

    // if rightWrist is higher than leftShoulder, put it next to leftEar
    if (rightWrist && leftShoulder && rightShoulder && rightElbow) {
      const { x, y } = rightWrist;
      const bodyWidth = leftShoulder.x - rightShoulder.x;
      const swordHeight = bodyWidth * 2;
      const swordWidth = bodyWidth / 2;
      const dBetweenWristAndElbow = rightWrist.x - rightElbow.x;
      const dBetweenWristAndElbowY = rightWrist.y - rightElbow.y;
      const startX = x;
      const startY = y;
      push();

      const slope = dBetweenWristAndElbowY / dBetweenWristAndElbow;

      const swordTilt = atan(slope);

      translate(startX, startY);

      if (dBetweenWristAndElbow > 0) {
        scale(-1.0, 1.0);
      }

      image(
        shusui,
        dBetweenWristAndElbow * 0.3,
        -swordHeight + dBetweenWristAndElbowY,
        swordWidth,
        swordHeight
      );
      pop();
    }
  }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255, 0, 0);
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255, 0, 0);
      line(
        partA.position.x,
        partA.position.y,
        partB.position.x,
        partB.position.y
      );
    }
  }
}
