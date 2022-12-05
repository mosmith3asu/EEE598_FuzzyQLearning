#!/usr/bin/env python3


import socket
HOST = '192.168.0.100'#raw_input("Enter IP address of UR controller:") # The remote host
PORT = 63350 # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
	print("Connecting to " + HOST)
	s.connect((HOST, PORT))
	f = open ("DataStream.csv", "w")
	try:
		print("Writing in DataStream.csv, Press ctrl + c to stop")
		while (1):
			data = s.recv(1024)
			print('data')
			#data = data.replace("(","")
			#data = data.replace(")","\n")
			#f.write(data)
	except KeyboardInterrupt:
		f.close
		s.close
except:
	print ("No connection")
