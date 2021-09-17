import ffmpeg as fp
import h5py


stream = fp.input("png/f%04.png")
stream = fp.output(stream, "video.mp4")
fp.run(stream)