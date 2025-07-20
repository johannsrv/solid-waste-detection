from flask import Flask, Response

from vido_stream.read_cam import Video

app = Flask(__name__)
vide_stream = Video(0)

@app.route('/video')
def video_feed():
    return Response(vide_stream.get_frames(),
                    mimetype='multipart/x-mixed-replace;  boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
