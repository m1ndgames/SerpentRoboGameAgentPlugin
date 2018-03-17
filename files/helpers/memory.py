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
	PlayerOne = c_int()
	PlayerTwo = c_int()
	ReadProcessMemory(processHandle, BaseAddress + 0x007DD82C, byref(game), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x304, byref(PlayerOne), 4, byref(numRead))
	ReadProcessMemory(processHandle, game.value + 0x310, byref(PlayerTwo), 4, byref(numRead))
	player1health = c_float()
	player2Health = c_float()
	ReadProcessMemory(processHandle, PlayerOne.value + 0xF8, byref(player1health), 4, byref(numRead))
	ReadProcessMemory(processHandle, PlayerTwo.value + 0xF8, byref(player2Health), 4, byref(numRead))
	return(player1health.value, player2Health.value)
