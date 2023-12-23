strng = "Human: What is my name? AI: ABCD Human: What is name of my father? AI: ABCD"
strng1 = strng.replace("AI:",",AI:").replace("Human:",",Human:").split(",")
strng1 = strng1[1:]
for i, message in enumerate(strng1):
    if i % 2 == 0:
      print(message)
    else:
      print(message)
