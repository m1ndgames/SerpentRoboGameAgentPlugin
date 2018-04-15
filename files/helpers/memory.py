from ctypes import *
from ctypes.wintypes import *
from time import sleep
import win32api
import ctypes, win32ui, win32process


def readhp ():
	OpenProcess = windll.kernel32.OpenProcess
	ReadProcessMemory = windll.kernel32.ReadProcessMemory
	FindWindowA = windll.user32.FindWindowA
	GetWindowThreadProcessId = windll.user32.GetWindowThreadProcessId

	PROCESS_ALL_ACCESS = 0x1F0FFF
	HWND = win32ui.FindWindow(None,u"Skullgirls Encore").GetSafeHwnd()
	PID = win32process.GetWindowThreadProcessId(HWND)[1]
	processHandle = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS,False,PID)

	#print(f"HWND: {HWND}")
	#print(f"PID: {PID}")
	#print(f"PROCESS: {processHandle}")

	# Open process for reading & writing:
	BaseAddress = win32process.EnumProcessModules(processHandle)[0]

	# Read out the app base (this is not the program / module base, the 'app' is just a huge object):
	appBase = c_int()
	numRead = c_int()

	# Read out player angle:
	game = c_int()
	PlayerOne_1 = c_int()
	PlayerOne_2 = c_int()
	PlayerOne_3 = c_int()
	PlayerTwo_1 = c_int()
	PlayerTwo_2 = c_int()
	PlayerTwo_3 = c_int()
	ReadProcessMemory(processHandle, BaseAddress + 0x007DD82C, byref(game), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x304, byref(PlayerOne_1), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x308, byref(PlayerOne_2), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x30C, byref(PlayerOne_3), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x310, byref(PlayerTwo_1), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x314, byref(PlayerTwo_2), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x318, byref(PlayerTwo_3), 4, byref(numRead))
	player1Health_1 = c_float()
	player1Health_2 = c_float()
	player1Health_3 = c_float()
	player2Health_1 = c_float()
	player2Health_2 = c_float()
	player2Health_3 = c_float()
	ReadProcessMemory(processHandle, PlayerOne_1.value + 0xF8, byref(player1Health_1), 4, byref(numRead))
	ReadProcessMemory(processHandle, PlayerOne_2.value + 0xF8, byref(player1Health_2), 4, byref(numRead))
	ReadProcessMemory(processHandle, PlayerOne_3.value + 0xF8, byref(player1Health_3), 4, byref(numRead))
	ReadProcessMemory(processHandle, PlayerTwo_1.value + 0xF8, byref(player2Health_1), 4, byref(numRead))
	ReadProcessMemory(processHandle, PlayerTwo_2.value + 0xF8, byref(player2Health_2), 4, byref(numRead))
	ReadProcessMemory(processHandle, PlayerTwo_3.value + 0xF8, byref(player2Health_3), 4, byref(numRead))
	p1hp_1 = int(player1Health_1.value)
	p1hp_2 = int(player1Health_2.value)
	p1hp_3 = int(player1Health_3.value)
	p2hp_1 = int(player2Health_1.value)
	p2hp_2 = int(player2Health_2.value)
	p2hp_3 = int(player2Health_3.value)
	
	if (p1hp_1 < 0):
		p1hp_1 = 0
	if (p1hp_2 < 0):
		p1hp_2 = 0
	if (p1hp_3 < 0):
		p1hp_3 = 0

	if (p2hp_1 < 0):
		p2hp_1 = 0
	if (p2hp_2 < 0):
		p2hp_2 = 0
	if (p2hp_3 < 0):
		p2hp_3 = 0

	p1_hp = (p1hp_1 + p1hp_2 + p1hp_3)
	p2_hp = (p2hp_1 + p2hp_2 + p2hp_3)

	return(p1_hp, p2_hp)
