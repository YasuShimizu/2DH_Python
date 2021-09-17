import subprocess
 
subprocess.call('ffmpeg -framerate 30 -i png/f%4d.png -r 60 -an -vcodec libx264 -pix_fmt yuv420p animation.mp4', shell=True)