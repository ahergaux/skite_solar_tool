# Source - https://stackoverflow.com/a/78273465
# Posted by Elikill58, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-29, License - CC BY-SA 4.0

import sys
import os
import chilkat
from logger import get_logger

log = get_logger("mail")


class iCloudConnection:

    def __init__(self, id="alexandre.hergaux", box="SkiteSolar"):
        log.info(f"Initializing iCloudConnection (user={id}, mailbox={box})")

        self.imap = chilkat.CkImap()

        self.imap.put_Ssl(True)
        self.imap.put_Port(993)
        log.debug("Connecting to imap.mail.me.com:993 (SSL)")
        success = self.imap.Connect("imap.mail.me.com")
        if not success:
            log.error(f"IMAP connection failed: {self.imap.lastErrorText()}")
            sys.exit()
        log.info("Connected to IMAP server (iCloud)")

        log.debug(f"Logging in as {id}")
        success = self.imap.Login(id, os.getenv("iCloudPassword"))
        if not success:
            log.error(f"IMAP login failed for {id}: {self.imap.lastErrorText()}")
            sys.exit()
        log.info("IMAP login successful")

        log.debug(f"Selecting mailbox: {box}")
        success = self.imap.SelectMailbox(box)
        if not success:
            log.error(f"Mailbox selection failed ({box}): {self.imap.lastErrorText()}")
            sys.exit()
        log.info(f"Mailbox '{box}' selected")

        log.info("iCloudConnection initialized")

    def get_all(self):
        count = self.imap.get_NumMessages()
        log.debug(f"Fetching {count} messages from mailbox")
        messages = []
        for i in range(1, count):
            msg = self.imap.FetchSingle(i, True)
            if msg is None:
                log.error(f"Failed to fetch message at index {i}")
            else:
                log.debug(f"Fetched message {i}: {getattr(msg, 'header', '(no header)')!r:.80}")
                messages.append(msg)
        log.info(f"Fetched {len(messages)} messages")
        return messages

    def move_to_inbox(self):
        log.debug("Moving all messages to Inbox")
        msgSet = self.imap.Search("ALL", True)
        result = self.imap.CopyMultiple(msgSet, "Inbox")
        if not result:
            log.error("move_to_inbox failed")
        else:
            log.info("Messages moved to Inbox")
        return result

    # TODO: missing delete for the mailbox

    def __delete__(self, instance):
        log.debug("Disconnecting IMAP")
        self.imap.Disconnect()
