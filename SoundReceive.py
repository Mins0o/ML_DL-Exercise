import serial
import time
from os import listdir

def ChooseDevice():
    devices = [dev for dev in listdir("/dev/") if ((dev[:4] == "ttyS" or dev[:4] == "ttyU"))]
    for deviceNumber in range(len(devices)):
        print(deviceNumber,end="\t")
        print(devices[deviceNumber])
    selection=int(input("Choose your serial device\n>>> "))
    while (selection<0 or selection>len(devices)):
        selection=int(input("Choose a valid number from the list\n>>> "))
    device = serial.Serial("/dev/"+devices[selection],230400)
    return device

def RecordData(device):
    fileName= input("Title name to save data ex)data1\n>>> ")+".tsv"
    with open("./"+fileName,'a') as dataFile:
       # Flush any incomplete transmission in the buffer.
        while (device.in_waiting>0):
            rawRead=device.read()
        # Loop and wait for input
        while True:
            rawRead=None
            if device.in_waiting>0:
                rawRead=device.read()
            else:
                time.sleep(0.5)
            # Checking the start of a transmission
            if(rawRead==b'\n'):
                if(device.read()==b'\n'):
                    # Two newlines are the starting mark
                    print("Recording")
                    pcmData=[]
                    intRead=0
                    while (not intRead==2570):
                        intRead=ReadTwoBytesToInt(device)
                        pcmData.append(intRead)
                    label=input("Label in one letter\nAdd another letter to finish recording\nYou can skip by not inputting any letters\n>>> ")
                    if len(label)==1:
                        for i in pcmData[:-2]: 
                            dataFile.write("{0},".format(i))
                        dataFile.write(str(pcmData[-2]))
                        dataFile.write("\t{0}\n".format(label))
                    elif len(label)>1: 
                        return
                    while device.in_waiting>0:
                        device.read()
                    pcmData=[]

def ReadTwoBytesToInt(device):
    return(int.from_bytes(device.read()+device.read(),"big"))

if __name__=="__main__":
    dev=ChooseDevice()
    RecordData(dev)
