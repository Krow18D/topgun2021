import requests
import time
import os

url = 'http://159.65.135.220/api'
def sendStatus():
  try:
    res = requests.post(url+'/status',{"status":"ok"})
    print(res)
  except:
    print("can't GET from server")
  time.sleep(1)
  try:
    res = requests.post(url+'/status',json = {"status":"ok"})
    print(res)
  except:
    print("can't GET from server")


# send small lime
# pls edit to pass arg with size,class
def detectSend(classes,size,imgName):
  if imgName != "": # edit size again
    multiFormdata = {
      'found' : classes,
      'size' : size,
      'img' : ("img" , open(imgName, 'rb'),'image/jpg')
    }
    # print(multiFormdata)
    res = requests.post(url+'/items', files = multiFormdata)
    print(res)
    if(imgName!=""):
      os.remove(imgName)
    # print("create lime")
  else:
    multiFormdata = {
      'found' : classes,
      'size' : size,
    }
    res = requests.post(url+'/items', multiFormdata)
    print(res)
    # print("create marker")

# sendStatus()
# detectSend("Lime","S","limecapture.jpg")
# detectSend(40)
