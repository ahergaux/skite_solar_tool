# Source - https://stackoverflow.com/a/78273465
# Posted by Elikill58, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-29, License - CC BY-SA 4.0

import sys
import os
import chilkat


class iCloudConnection:
   
    def __init__(self, id = "alexandre.hergaux", box = "SkiteSolar"):
        self.imap = chilkat.CkImap()

        # Connect to the iCloud IMAP Mail Server
        self.imap.put_Ssl(True)
        self.imap.put_Port(993)
        success = self.imap.Connect("imap.mail.me.com")
        if (success != True):
            print(self.imap.lastErrorText())
            sys.exit()

        # The username is usually the name part of your iCloud email address 
        # (for example, emilyparker, not emilyparker@icloud.com).
        success = self.imap.Login(id,os.getenv("iCloudPassword"))
        if (success != True):
            print(self.imap.lastErrorText())
            sys.exit()

        # Select an IMAP folder/mailbox
        success = self.imap.SelectMailbox(box)
        if (success != True):
            print(self.imap.lastErrorText())
            sys.exit()

    def get_all(self):
        return [self.imap.FetchSingle(1, True) for i in range(1, self.imap.get_NumMessages())]            
            
    def move_to_inbox(self):
        msgSet = self.imap.Search("ALL", True)
        return self.imap.CopyMultiple(msgSet,"Inbox")
    # TODO: missing delete for the mailbox
        
    def __delete__(self, instance):
        self.imap.Disconnect()

