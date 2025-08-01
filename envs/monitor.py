import subprocess as sp

class Monitor:
    def __init__(self, width, height, saved_path):
        self.command = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", "60",
            "-i", "-", "-an", "-vcodec", "mpeg4", saved_path
        ]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            self.pipe = None

    def record(self, image_array):
        if self.pipe:
            self.pipe.stdin.write(image_array.tobytes())
            
    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()