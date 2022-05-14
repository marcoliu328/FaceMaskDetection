try:
    from numpy import broadcast
    from flask import Flask, render_template, Response, request
    from tensorflow.keras.models import load_model
    from flask_socketio import SocketIO
    from flask_socketio import emit
    from imutils.video import VideoStream
    import os 
    import sys
    import json
    import cv2
    import imutils
    import detect_mask
except Exception as e:
    print("Missing Module: {}".format(e))

switch = 0
app = Flask(__name__)
socketio = SocketIO(app)
camera = None

#load face_detector model (architecture definition and weights)
definitionPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceModel = cv2.dnn.readNet(definitionPath, weightsPath)

#load our mask_detector model from training
maskModel = load_model("face_mask_detection_model")

def gen():
    while True:
        if switch == 1 and camera is not None:
            success, frame = camera.read()
            if success:
                frame = imutils.resize(frame, width = 640, height = 480)
                #send frame and both models for prediction
                (locs,preds) = detect_mask.detect_and_predict(frame, faceModel, maskModel)

                #analyze detected face locations
                for (box, pred) in zip(locs, preds):

                    #grab coordinates and prediction, set label and colour (BGR)
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred  

                    if mask > withoutMask:
                        label = "Mask"
                        color = (0, 255, 0) #Green
                    else:
                        label = "No Mask"
                        color = (0, 0, 255) #Red

                    #Add the probability of label to the label (floating point)
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    #output the label and bounding box
                    cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness=2)

                success, jpg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

            else: 
                break
        
        else: 
            break

@socketio.on('connect')
def connect():
    try:
        print("connected", file=sys.stderr)
    except Exception as e:
        print("connect socket failed", file=sys.stderr)
        print(e)

@socketio.on('disconnect')
def disconnect():
    global switch, camera
    try:
        print("disconnected", file=sys.stderr)
        if switch == 1:
            camera.release()
            cv2.destroyAllWindows()
            switch = 0
            camera = None
    except Exception as e:
        print("did not disconnect", file=sys.stderr)
        print(e)
        pass
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/requests', methods = ['POST', 'GET'])
def requests():
    global switch, camera
    if request.method == 'POST':
        if "start" in request.form and switch == 0:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            switch = 1
            print("starting")
        elif "stop" in request.form and switch == 1:
            try:
                camera.release()
                cv2.destroyAllWindows()
                switch = 0
                camera = None
                print("stopping")
            except Exception as e:
                print(e)

        print(switch)


    elif request.method == "GET":
        return render_template('index.html', data=switch)

    return render_template('index.html', data=switch)


if __name__ == '__main__':
    socketio.run(app)
    #app.run(host="0.0.0.0", port=8080)