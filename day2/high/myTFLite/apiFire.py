import requests
import time


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
def detectSend(classes,):
  if size < 39: # edit size again
    imgUrl = '/Users/krxw/Me/TGR15/im/'
    multiFormdata = {
      'found' : "Lime",
      'size' : "S",
      'img' : ('emi.jpg' , open('/Users/krxw/Me/TGR15/im/emi.jpg', 'rb'),'image/jpg'),
    }
    requests.post(url+'/items', multiFormdata)
    # print("create lime")
  else:
    multiFormdata = {
      'found' : "Marker",
      'size' : "L",
    }
    requests.post(url+'/items', multiFormdata)
    # print("create marker")

sendStatus()
# detectSend(37)
# detectSend(40)
