
# # OBJECT DETECTION AND PEOPLE COUTER
"""
    DEMO TENSORFLOW OBJECT DETECTION API
    LIENG Hongky ,lieng.hongky@gmail.com, 2018
"""

# # Imports
import telepot
import datetime
import threading
import time
class TelegramSender(object):

    picPath = None
    token = 'TELEGRAM_BOT_API_TOKEN'
    bot = None
    isText = True
    text = ''
    recieverID = "492169246"
    
    def __init__(self, interval=1):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.interval = interval
        self.bot = telepot.Bot(self.token)

    def sendText(self):
      chat_id = self.recieverID
      try :
        self.bot.sendMessage(chat_id,self.text)
      except:
        print('cannot send tex')
    def sendPicture(self):

        
        chat_id = self.recieverID
        try :
          print(self.picPath)
          pic = open(self.picPath,'rb')
          self.bot.sendPhoto(chat_id, pic, self.text)
        except:
          print('cannot send pic')
    def startText(self,text):
        self.isText = True
        self.text = text
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start() 
    def start(self,path,text):
        self.isText = False
        self.text = text
        self.picPath = path
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start() 
    def run(self):
        if self.isText == True:
          self.sendText()
        else:
          self.sendPicture()
######################################ENDCLASS