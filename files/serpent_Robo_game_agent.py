from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
from serpent.sprite_locator import SpriteLocator
from serpent.frame_grabber import FrameGrabber
from serpent.config import config
import serpent.utilities
from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace
sprite_locator = SpriteLocator()
import serpent.cv
import time
import cv2
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.measure
import skimage.draw
import skimage.segmentation
import skimage.color
import os
import gc
from datetime import datetime
import collections
import numpy as np
from .helpers.memory import readhp
from colorama import init
init()
from colorama import Fore, Style

class SerpentRoboGameAgent(GameAgent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		self.frame_handlers["PLAY"] = self.handle_play

		self.frame_handler_setups["PLAY"] = self.setup_play

		self.sprite_locator = SpriteLocator()

		self.game_state = None
		self._reset_game_state()

	def setup_play(self):
		input_mapping = {
			"W": [KeyboardKey.KEY_W],
			"A": [KeyboardKey.KEY_A],
			"S": [KeyboardKey.KEY_S],
			"D": [KeyboardKey.KEY_D],
			"WA": [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
			"WD": [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
			"SA": [KeyboardKey.KEY_S, KeyboardKey.KEY_A],
			"SD": [KeyboardKey.KEY_S, KeyboardKey.KEY_D],
			"J": [KeyboardKey.KEY_J],
			"K": [KeyboardKey.KEY_K],
			"L": [KeyboardKey.KEY_L],
			"U": [KeyboardKey.KEY_U],
			"I": [KeyboardKey.KEY_I],
			"O": [KeyboardKey.KEY_O],
			"JU": [KeyboardKey.KEY_J, KeyboardKey.KEY_U],
			"KI": [KeyboardKey.KEY_K, KeyboardKey.KEY_I],
			"LO": [KeyboardKey.KEY_L, KeyboardKey.KEY_O],
			"N": [KeyboardKey.KEY_N],
			"M": [KeyboardKey.KEY_M],
			"NONE": []
		}

		self.key_mapping = {
			KeyboardKey.KEY_W.name: "MOVE UP",
			KeyboardKey.KEY_A.name: "MOVE LEFT",
			KeyboardKey.KEY_S.name: "MOVE DOWN",
			KeyboardKey.KEY_D.name: "MOVE RIGHT",
			KeyboardKey.KEY_J.name: "LIGHT PUNCH",
			KeyboardKey.KEY_K.name: "MEDIUM PUNCH",
			KeyboardKey.KEY_L.name: "HARD PUNCH",
			KeyboardKey.KEY_U.name: "LIGHT KICK",
			KeyboardKey.KEY_I.name: "MEDIUM KICK",
			KeyboardKey.KEY_O.name: "HARD KICK",
			KeyboardKey.KEY_N.name: "START",
			KeyboardKey.KEY_M.name: "SELECT"
		}

		movement_action_space = KeyboardMouseActionSpace(
			directional_keys=["W", "A", "S", "D", "WA", "WD", "SA", "SD", "NONE"]
		)

		fightinput_action_space = KeyboardMouseActionSpace(
			fightinput_keys=["J", "K", "L", "U", "I", "O", "JU", "KI", "LO", "NONE"]
		)

		movement_model_file_path = "datasets/fighting_movement_dqn_0_1_.h5".replace("/", os.sep)
		self.dqn_movement = DDQN(
			model_file_path=movement_model_file_path if os.path.isfile(movement_model_file_path) else None,
			input_shape=(100, 100, 4),
			input_mapping=input_mapping,
			action_space=movement_action_space,
			replay_memory_size=5000,
			max_steps=1000000,
			observe_steps=1000,
			batch_size=32,
			initial_epsilon=1,
			final_epsilon=0.01,
			override_epsilon=False
		)

		fightinput_model_file_path = "datasets/fighting_fightinput_dqn_0_1_.h5".replace("/", os.sep)
		self.dqn_fightinput = DDQN(
			model_file_path=fightinput_model_file_path if os.path.isfile(fightinput_model_file_path) else None,
			input_shape=(100, 100, 4),
			input_mapping=input_mapping,
			action_space=fightinput_action_space,
			replay_memory_size=5000,
			max_steps=1000000,
			observe_steps=1000,
			batch_size=32,
			initial_epsilon=1,
			final_epsilon=0.01,
			override_epsilon=False
		)
		print("Debug: Game Started")

	def handle_play(self, game_frame):
		#print("Debug: Main")
		title_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_TITLE_TEXT'], game_frame=game_frame)
		menu_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_MAINMENU_TEXT'], game_frame=game_frame)
		fightmenu_select_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_FIGHTMENU_SELECT'], game_frame=game_frame)
		playerselect_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_PLAYERSELECT'], game_frame=game_frame)
		backbutton_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_BACKBUTTON'], game_frame=game_frame)
		fightcheck_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_FIGHTCHECK'], game_frame=game_frame)
		roundstart_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_ROUNDSTART'], game_frame=game_frame)
		retrybutton_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_FIGHTMENU_RETRY'], game_frame=game_frame)
		backbutton_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_BACKBUTTON'], game_frame=game_frame)

		(self.p1hp, self.p2hp) = readhp()
		self.game_state["health"].appendleft(self.p1hp)
		self.game_state["enemy_health"].appendleft(self.p2hp)
		
		if (roundstart_locator):
			#print("Debug: roundstart_locator Locator")
			self.game_state["fightstarted"] = True
		elif (retrybutton_locator):
			#print("Debug: retrybutton_locator Locator")
			self.handle_fight_end(game_frame)
		elif (fightcheck_locator):
			#print("Debug: fightcheck_locator Locator")
			self.handle_fight(game_frame)
		elif (title_locator):
			#print("Debug: title_locator Locator")
			self.handle_menu_title(game_frame)
		elif (menu_locator):
			#print("Debug: menu_locator Locator")
			self.handle_menu_select(game_frame)
		elif (playerselect_locator):
			#print("Debug: playerselect_locator Locator")
			self.handle_player_select(game_frame)
		elif (backbutton_locator):
			#print("Debug: backbutton_locator Locator")
			self.handle_backbutton(game_frame)
		elif ((fightmenu_select_locator) and (self.game_state["current_run"] != 1)):
			#print("Debug: fightmenu_select_locator Locator")
			self.handle_fightmenu_select(game_frame)
		else:
			return

	def handle_retry_button(self, game_frame):
		if (self.game_state["current_run"] % 25 == 0):
			print(Fore.RED + 'Changing Opponent')
			print(Style.RESET_ALL)
			time.sleep(1)
			self.input_controller.tap_key(KeyboardKey.KEY_S)
			time.sleep(0.5)
			self.input_controller.tap_key(KeyboardKey.KEY_J)
			time.sleep(1)
		else:
			print(Fore.RED + 'Restarting Fight')
			print(Style.RESET_ALL)
			time.sleep(1)
			self.input_controller.tap_key(KeyboardKey.KEY_J)
			time.sleep(1)

	def handle_backbutton(self, game_frame):
		print(Fore.RED + 'Pressing Select')
		print(Style.RESET_ALL)
		self.input_controller.tap_key(KeyboardKey.KEY_M)
		time.sleep(1)

	def handle_menu_title(self, game_frame):
		print(Fore.RED + 'Pressing Start')
		print(Style.RESET_ALL)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(2)

	def handle_fightmenu_select(self, game_frame):
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(2)

	def handle_player_select(self, game_frame):
		time.sleep(1)
		print(Fore.RED + 'Choosing one Char')
		self.input_controller.tap_key(KeyboardKey.KEY_A)
		time.sleep(0.3)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(0.5)
		print("Choosing Robo")
		self.input_controller.tap_key(KeyboardKey.KEY_S)
		time.sleep(0.3)
		self.input_controller.tap_key(KeyboardKey.KEY_S)
		time.sleep(0.3)
		self.input_controller.tap_key(KeyboardKey.KEY_D)
		time.sleep(0.3)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(0.5)
		print("Choosing one CPU Char")
		self.input_controller.tap_key(KeyboardKey.KEY_A)
		time.sleep(0.3)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(0.3)
		print("Choosing Random CPU Char")
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(0.3)
		print("Starting Game")
		print(Style.RESET_ALL)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(1)


	def handle_menu_select(self, game_frame):
		menu_selector = sprite_locator.locate(sprite=self.game.sprites['SPRITE_MAINMENU_SINGLEPLAY'], game_frame=game_frame)
		if (menu_selector):
			print(Fore.RED + 'Starting Singleplayer Mode')
			print(Style.RESET_ALL)
			self.input_controller.tap_key(KeyboardKey.KEY_J)
			time.sleep(1)
			self.input_controller.tap_key(KeyboardKey.KEY_S)
			time.sleep(1)
			self.input_controller.tap_key(KeyboardKey.KEY_S)
			time.sleep(1)
			self.input_controller.tap_key(KeyboardKey.KEY_J)
			time.sleep(1)
		else:
			self.input_controller.tap_key(KeyboardKey.KEY_S)
			time.sleep(1)

	def handle_fight(self, game_frame):
		gc.disable()

		if not (self.game_state["fightstarted"]):
			return

		if ((self.game_state["health"][0] == 0) and (self.game_state["health"][1] == 0) or (self.game_state["enemy_health"][0] == 0) and (self.game_state["enemy_health"][1] == 0)):
			return

		if self.dqn_movement.first_run:
			self.dqn_movement.first_run = False
			self.dqn_fightinput.first_run = False
			return None

		if self.dqn_movement.frame_stack is None:
			pipeline_game_frame = FrameGrabber.get_frames(
				[0],
				frame_shape=(self.game.frame_height, self.game.frame_width),
				frame_type="PIPELINE",
				dtype="float64"
			).frames[0]

			self.dqn_movement.build_frame_stack(pipeline_game_frame.frame)
			self.dqn_fightinput.frame_stack = self.dqn_movement.frame_stack
		else:
			game_frame_buffer = FrameGrabber.get_frames(
				[0, 4, 8, 12],
				frame_shape=(self.game.frame_height, self.game.frame_width),
				frame_type="PIPELINE",
				dtype="float64"
			)

			if self.dqn_movement.mode == "TRAIN":
				reward_movement, reward_fightinput = self._calculate_reward()

				self.game_state["run_reward_movement"] += reward_movement
				self.game_state["run_reward_fightinput"] += reward_fightinput

				self.dqn_movement.append_to_replay_memory(
					game_frame_buffer,
					reward_movement,
					terminal=self.game_state["health"] == 0
				)

				self.dqn_fightinput.append_to_replay_memory(
					game_frame_buffer,
					reward_fightinput,
					terminal=self.game_state["health"] == 0
				)

				# Every 2000 steps, save latest weights to disk
				if self.dqn_movement.current_step % 2000 == 0:
					self.dqn_movement.save_model_weights(
						file_path_prefix=f"datasets/fighting_movement"
					)

					self.dqn_fightinput.save_model_weights(
						file_path_prefix=f"datasets/fighting_fightinput"
					)

				# Every 20000 steps, save weights checkpoint to disk
				if self.dqn_movement.current_step % 20000 == 0:
					self.dqn_movement.save_model_weights(
						file_path_prefix=f"datasets/fighting_movement",
						is_checkpoint=True
					)

					self.dqn_fightinput.save_model_weights(
						file_path_prefix=f"datasets/fighting_fightinput",
						is_checkpoint=True
					)
			elif self.dqn_movement.mode == "RUN":
				self.dqn_movement.update_frame_stack(game_frame_buffer)
				self.dqn_fightinput.update_frame_stack(game_frame_buffer)

			run_time = datetime.now() - self.started_at
			serpent.utilities.clear_terminal()
			print("")
			print(Fore.YELLOW)
			print(Style.BRIGHT)
			print(f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
			print(Style.RESET_ALL)
			print("")
			print(Fore.GREEN)
			print(Style.BRIGHT)
			print("MOVEMENT NEURAL NETWORK:\n")
			self.dqn_movement.output_step_data()
			print("")
			print("FIGHT NEURAL NETWORK:\n")
			self.dqn_fightinput.output_step_data()
			print(Style.RESET_ALL)
			print("")
			print(Style.BRIGHT)
			print(f"CURRENT RUN: {self.game_state['current_run']}")
			print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")
			print("")
			print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward_movement'] + self.game_state['run_reward_fightinput'], 4)}")
			print(f"COMBO MULTIPLICATOR: {self.game_state['multiplier_damage']}")
			print(f"CURRENT HEALTH: {self.game_state['health'][0]}")
			print(f"CURRENT ENEMY HEALTH: {self.game_state['enemy_health'][0]}")
			print("")
			print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
			print(Style.RESET_ALL)

			self.dqn_movement.pick_action()
			self.dqn_movement.generate_action()
	
			self.dqn_fightinput.pick_action(action_type=self.dqn_movement.current_action_type)
			self.dqn_fightinput.generate_action()

			movement_keys = self.dqn_movement.get_input_values()
			fightinput_keys = self.dqn_fightinput.get_input_values()

			print("")
			print(Fore.GREEN)
			print(Style.BRIGHT)
			print("" + " + ".join(list(map(lambda k: self.key_mapping.get(k.name), movement_keys + fightinput_keys))))
			print(Style.RESET_ALL)
	
			self.input_controller.handle_keys(movement_keys + fightinput_keys)

			if self.dqn_movement.current_action_type == "PREDICTED":
				self.game_state["run_predicted_actions"] += 1

			self.dqn_movement.erode_epsilon(factor=2)
			self.dqn_fightinput.erode_epsilon(factor=2)

			self.dqn_movement.next_step()
			self.dqn_fightinput.next_step()

			self.game_state["current_run_steps"] += 1

	def _reset_game_state(self):
		self.game_state = {
			"health": collections.deque(np.full((8,), 6), maxlen=8),
			"enemy_health": collections.deque(np.full((8,), 654), maxlen=8),
			"current_run": 1,
			"current_run_steps": 0,
			"run_reward_movement": 0,
			"run_reward_fightinput": 0,
			"run_future_rewards": 0,
			"run_predicted_actions": 0,
			"run_timestamp": datetime.utcnow(),
			"last_run_duration": 0,
			"record_time_alive": dict(),
			"record_enemy_hp": dict(),
			"random_time_alive": None,
			"random_time_alives": list(),
			"random_enemy_hp": None,
			"random_enemy_hps": list(),
			"fightstarted": None,
			"multiplier_damage": 0
		}

	def _calculate_reward(self):
		reward_movement = 0
		reward_fightinput = 0

		# getting hit by enemy
		if self.game_state["health"][0] < self.game_state["health"][1]:
			self.game_state["multiplier_damage"] = 0
			reward_movement += -0.10
			reward_fightinput += -0.10
		else:
			reward_movement += 0.05

		# hitting the enemy
		if self.game_state["enemy_health"][0] < self.game_state["enemy_health"][1]:
			# combo multiplicator
			self.game_state["multiplier_damage"] += 0.20
			if self.game_state["multiplier_damage"] > 1:
				self.game_state["multiplier_damage"] = 1
				
			# check how much dmg the attack did and add 0.05 per class to reward
			if (self.game_state["enemy_health"][1] - self.game_state["enemy_health"][0]) > 150: # light
				reward_fightinput += 0.05
			if (self.game_state["enemy_health"][1] - self.game_state["enemy_health"][0]) > 500: # medium
				reward_fightinput += 0.05
			if (self.game_state["enemy_health"][1] - self.game_state["enemy_health"][0]) > 750: # hard
				reward_fightinput += 0.05
			
			# calculate reward
			reward_fightinput += (1 * self.game_state["multiplier_damage"])
		else:
			reward_fightinput += -0.05
			reward_movement += -0.01

		# enemy wasnt hit for 5 rounds
		if self.game_state["enemy_health"][0] == self.game_state["enemy_health"][5]:
			self.game_state["multiplier_damage"] = 0

		# return rewards
		return reward_movement, reward_fightinput

	def handle_fight_end(self, game_frame):
		self.game_state["fightstarted"] = None
		self.input_controller.handle_keys([])
		self.game_state["current_run"] += 1
		self.handle_fight_training(game_frame)

	def handle_fight_training(self, game_frame):
		serpent.utilities.clear_terminal()
		timestamp = datetime.utcnow()
		timestamp_delta = timestamp - self.game_state["run_timestamp"]
		self.game_state["last_run_duration"] = timestamp_delta.seconds
		gc.enable()
		gc.collect()
		gc.disable()

		if self.dqn_movement.mode in ["TRAIN", "RUN"]:
			# Check for Records
			if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
				self.game_state["record_time_alive"] = {
					"value": self.game_state["last_run_duration"],
					"run": self.game_state["current_run"],
					"predicted": self.dqn_movement.mode == "RUN",
					"enemy_hp": self.game_state["enemy_health"][0]
				}

			if self.game_state["enemy_health"][0] < self.game_state["record_enemy_hp"].get("value", 1000):
				self.game_state["record_enemy_hp"] = {
					"value": self.game_state["enemy_health"][0],
					"run": self.game_state["current_run"],
					"predicted": self.dqn_movement.mode == "RUN",
					"time_alive": self.game_state["last_run_duration"]
				}
		else:
			self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
			self.game_state["random_enemy_hps"].append(self.game_state["enemy_health"][0])

			self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])
			self.game_state["random_enemy_hp"] = np.mean(self.game_state["random_enemy_hps"])

		self.game_state["current_run_steps"] = 0

		self.input_controller.handle_keys([])

		if self.dqn_movement.mode == "TRAIN":
			for i in range(16):
				serpent.utilities.clear_terminal()
				print("")
				print(Fore.GREEN)
				print(Style.BRIGHT)
				print(f"TRAINING ON MINI-BATCHES: {i + 1}/16")
				print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 25 == 0 else ''}")
				print(Style.RESET_ALL)
				
				self.dqn_movement.train_on_mini_batch()
				self.dqn_fightinput.train_on_mini_batch()

		self.game_state["run_timestamp"] = datetime.utcnow()
		self.game_state["run_reward_movement"] = 0
		self.game_state["run_reward_fightinput"] = 0
		self.game_state["run_predicted_actions"] = 0
		self.game_state["health"] = collections.deque(np.full((8,), 6), maxlen=8)
		self.game_state["enemy_health"] = collections.deque(np.full((8,), 654), maxlen=8)

		if self.dqn_movement.mode in ["TRAIN", "RUN"]:
			if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
				self.dqn_movement.update_target_model()
				self.dqn_fightinput.update_target_model()

			if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
				self.dqn_movement.enter_run_mode()
				self.dqn_fightinput.enter_run_mode()
			else:
				self.dqn_movement.enter_train_mode()
				self.dqn_fightinput.enter_train_mode()


		self.handle_retry_button(game_frame)
