from serpent.game_agent import GameAgent

from serpent.frame_grabber import FrameGrabber
from serpent.sprite_locator import SpriteLocator
from serpent.input_controller import KeyboardKey
sprite_locator = SpriteLocator()

from serpent.config import config
#from serpent.visual_debugger.visual_debugger import VisualDebugger
import serpent.cv

from .helpers.terminal_printer import TerminalPrinter
from .helpers.ppo import SerpentPPO

import itertools
import collections

import time
import os
import pickle
import subprocess
import shlex
import random
import numpy as np

import skimage.io
import skimage.filters
import skimage.morphology
import skimage.measure
import skimage.draw
import skimage.segmentation
import skimage.color

from datetime import datetime
#from .helpers.frame_processing import readhp
from .helpers.memory import readhp

class SerpentRoboGameAgent(GameAgent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.frame_handlers["PLAY"] = self.handle_play
		self.frame_handler_setups["PLAY"] = self.setup_play
		self.sprite_locator = SpriteLocator()
		self.game_state = None
		self.printer = TerminalPrinter()
 
	def setup_play(self):
		self.first_run = True

		self.enemy = "P2"
		self.enemy_hp_mapping = {
			"P2": 15500
		}

		move_inputs = {
			"MOVE UP": [KeyboardKey.KEY_W],
			"MOVE LEFT": [KeyboardKey.KEY_A],
			"MOVE DOWN": [KeyboardKey.KEY_S],
			"MOVE RIGHT": [KeyboardKey.KEY_D],
			"MOVE TOP-LEFT": [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
			"MOVE TOP-RIGHT": [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
			"MOVE DOWN-LEFT": [KeyboardKey.KEY_S, KeyboardKey.KEY_A],
			"MOVE DOWN-RIGHT": [KeyboardKey.KEY_S, KeyboardKey.KEY_D],
			"DON'T MOVE": []
		}

		shoot_inputs = {
			"LP": [KeyboardKey.KEY_U],
			"MP": [KeyboardKey.KEY_I],
			"HP": [KeyboardKey.KEY_O],
			"LK": [KeyboardKey.KEY_J],
			"MK": [KeyboardKey.KEY_K],
			"HK": [KeyboardKey.KEY_L],
			"LPLK": [KeyboardKey.KEY_U, KeyboardKey.KEY_J],
			"MPMK": [KeyboardKey.KEY_I, KeyboardKey.KEY_K],
			"HPHK": [KeyboardKey.KEY_O, KeyboardKey.KEY_L],
			"DON'T FIGHT": []
		}

		self.game_inputs = dict()
		for move_label, shoot_label in itertools.product(move_inputs, shoot_inputs):
			label = f"{move_label.ljust(20)}{shoot_label}"
			self.game_inputs[label] = move_inputs[move_label] + shoot_inputs[shoot_label]

		self.run_count = 0
		self.run_reward = 0
		self.fightcount = 1

		self.observation_count = 0
		self.episode_observation_count = 0

		self.performed_inputs = collections.deque(list(), maxlen=8)

		self.reward_10 = collections.deque(list(), maxlen=10)
		self.reward_100 = collections.deque(list(), maxlen=100)
		self.reward_1000 = collections.deque(list(), maxlen=1000)

		self.rewards = list()

		self.average_reward_10 = 0
		self.average_reward_100 = 0
		self.average_reward_1000 = 0

		self.top_reward = 0
		self.top_reward_run = 0

		self.previous_enemy_hp = self.enemy_hp_mapping[self.enemy]

		self.enemy_hp_10 = collections.deque(list(), maxlen=10)
		self.enemy_hp_100 = collections.deque(list(), maxlen=100)
		self.enemy_hp_1000 = collections.deque(list(), maxlen=1000)

		self.average_enemy_hp_10 = self.enemy_hp_mapping[self.enemy]
		self.average_enemy_hp_100 = self.enemy_hp_mapping[self.enemy]
		self.average_enemy_hp_1000 = self.enemy_hp_mapping[self.enemy]

		self.best_enemy_hp = self.enemy_hp_mapping[self.enemy]
		self.best_enemy_hp_run = 0

		self.death_check = False
		self.just_relaunched = False
		self.fightstarted = False
		self.frame_buffer = None

		self.take = 1
		
		self.ppo_agent = SerpentPPO(
			frame_shape=(100, 100, 4),
			game_inputs=self.game_inputs
		)

		try:
			self.ppo_agent.agent.restore_model(directory=os.path.join(os.getcwd(), "datasets", "skullgirls"))
			self.restore_metadata()
		except Exception:
			pass

		self.analytics_client.track(event_key="INITIALIZE", data=dict(episode_rewards=[]))

		for reward in self.rewards:
			self.analytics_client.track(event_key="EPISODE_REWARD", data=dict(reward=reward))
			time.sleep(0.01)

		# Warm Agent?
		game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
		self.ppo_agent.generate_action(game_frame_buffer)

		self.health = collections.deque(np.full((16,), 6), maxlen=16)
		self.enemy_health = collections.deque(np.full((8,), self.enemy_hp_mapping[self.enemy]), maxlen=8)

		self.multiplier_damage = 0

		self.enemy_skull_image = None

		self.started_at = datetime.utcnow().isoformat()
		self.episode_started_at = None

		self.paused_at = None

	def handle_play(self, game_frame):
		# various sprite locators
		title_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_TITLE_TEXT'], game_frame=game_frame)
		menu_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_MAINMENU_TEXT'], game_frame=game_frame)
		fightmenu_select_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_FIGHTMENU_SELECT'], game_frame=game_frame)
		playerselect_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_PLAYERSELECT'], game_frame=game_frame)
		backbutton_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_BACKBUTTON'], game_frame=game_frame)
		fightcheck_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_FIGHTCHECK'], game_frame=game_frame)
		roundstart_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_ROUNDSTART'], game_frame=game_frame)
		retrybutton_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_FIGHTMENU_RETRY'], game_frame=game_frame)
		backbutton_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_BACKBUTTON'], game_frame=game_frame)

		# reads hp from frame
		p1hp_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["P1_HP"])
		p2hp_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["P2_HP"])
		(self.p1hp, self.p2hp) = readhp()
		self.health.appendleft(self.p1hp)
		self.enemy_health.appendleft(self.p2hp)
		reward, is_alive, enemy_dead = self.reward_skullgirls([None, None, game_frame, None])
		
		# Defines current take in fight round
		take2_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_TAKE2'], game_frame=game_frame)
		take3_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_TAKE3'], game_frame=game_frame)
		if (take2_locator):
			self.take = 2
		elif (take3_locator):
			self.take = 3
			
		if (roundstart_locator):
			#print("Debug: roundstart_locator Locator")
			self.fightstarted = True
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
		elif ((fightmenu_select_locator) and (self.fightcount != 1)):
			#print("Debug: fightmenu_select_locator Locator")
			self.handle_fightmenu_select(game_frame)
		else:
			return

	def handle_retry_button(self, game_frame):
		print("Pressing LP")
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(1)

	def handle_backbutton(self, game_frame):
		print("Pressing Select")
		self.input_controller.tap_key(KeyboardKey.KEY_M)
		time.sleep(1)

	def handle_menu_title(self, game_frame):
		print("Pressing Start")
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(2)

	def handle_fightmenu_select(self, game_frame):
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(2)

	def handle_player_select(self, game_frame):
		print("Choosing one Char")
		self.input_controller.tap_key(KeyboardKey.KEY_A)
		time.sleep(1)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(1)
		print("Choosing Robo")
		self.input_controller.tap_key(KeyboardKey.KEY_S)
		time.sleep(1)
		self.input_controller.tap_key(KeyboardKey.KEY_S)
		time.sleep(1)
		self.input_controller.tap_key(KeyboardKey.KEY_D)
		time.sleep(1)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(1)
		print("Choosing one CPU Char")
		self.input_controller.tap_key(KeyboardKey.KEY_A)
		time.sleep(1)
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(1)
		print("Choosing Random CPU Char")
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(2)
		print("Starting Game")
		self.input_controller.tap_key(KeyboardKey.KEY_J)
		time.sleep(2)

	def handle_menu_select(self, game_frame):
		menu_selector = sprite_locator.locate(sprite=self.game.sprites['SPRITE_MAINMENU_SINGLEPLAY'], game_frame=game_frame)
		if (menu_selector):
			print("Starting Singleplayer Mode")
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
		timeout_locator = sprite_locator.locate(sprite=self.game.sprites['SPRITE_TIMEOUT'], game_frame=game_frame)
		if (timeout_locator):
			return

		if (self.fightstarted == False):
			return

		if ((self.health[0] < 0.0) and (self.health[1] < 0.0) or (self.enemy_health[1] < 0.0) and (self.enemy_health[1] < 0.0)):
			return
			
		reward, is_alive, enemy_dead = self.reward_skullgirls([None, None, game_frame, None])
		self.printer.add("")
		self.printer.add(f"Stage Started At: {self.started_at}")
		self.printer.add("")
		if self.frame_buffer is not None:
			self.run_reward += reward

			self.observation_count += 1
			self.episode_observation_count += 1

			self.analytics_client.track(event_key="RUN_REWARD", data=dict(reward=reward))

			episode_over = self.episode_observation_count > (120 * config["SerpentSkullgirlsGamePlugin"]["fps"])


			self.ppo_agent.observe(reward, terminal=(not is_alive or enemy_dead))

		self.printer.add(f"Observation Count: {self.observation_count}")
		self.printer.add(f"Episode Observation Count: {self.episode_observation_count}")
		self.printer.add(f"Current Batch Size: {self.ppo_agent.agent.batch_count}")
		self.printer.add(f"Current Round: {self.fightcount}")
		self.printer.add(f"Take: {self.take}")
		self.printer.add(f"HP P1: {int(self.p1hp)}")
		self.printer.add(f"HP P2: {int(self.p2hp)}")
		self.printer.add("")
		self.printer.add(f"Run Reward: {round(self.run_reward, 2)}")
		self.printer.add("")
		self.printer.add(f"Average Rewards (Last 10 Runs): {round(self.average_reward_10, 2)}")
		self.printer.add(f"Average Rewards (Last 100 Runs): {round(self.average_reward_100, 2)}")
		self.printer.add(f"Average Rewards (Last 1000 Runs): {round(self.average_reward_1000, 2)}")
		self.printer.add("")
		self.printer.add(f"Top Run Reward: {round(self.top_reward, 2)} (Run #{self.top_reward_run})")
		self.printer.add("")
		enemy_hp_percent = round((self.average_enemy_hp_10 / self.enemy_hp_mapping[self.enemy]) * 100.0, 2)
		self.printer.add(f"Average Enemy HP (Last 10 Runs): {round(self.average_enemy_hp_10, 2)} / {self.enemy_hp_mapping[self.enemy]} ({enemy_hp_percent}% left)")
		enemy_hp_percent = round((self.average_enemy_hp_100 / self.enemy_hp_mapping[self.enemy]) * 100.0, 2)
		self.printer.add(f"Average Enemy HP (Last 100 Runs): {round(self.average_enemy_hp_100, 2)} / {self.enemy_hp_mapping[self.enemy]} ({enemy_hp_percent}% left)")
		enemy_hp_percent = round((self.average_enemy_hp_1000 / self.enemy_hp_mapping[self.enemy]) * 100.0, 2)
		self.printer.add(f"Average Enemy HP (Last 1000 Runs): {round(self.average_enemy_hp_1000, 2)} / {self.enemy_hp_mapping[self.enemy]} ({enemy_hp_percent}% left)")
		self.printer.add("")
		enemy_hp_percent = round((self.previous_enemy_hp / self.enemy_hp_mapping[self.enemy]) * 100.0, 2)
		self.printer.add(f"Previous Run Enemy HP: {round(self.previous_enemy_hp, 2)} / {self.enemy_hp_mapping[self.enemy]} ({enemy_hp_percent}% left)")
		self.printer.add("")
		enemy_hp_percent = round((self.best_enemy_hp / self.enemy_hp_mapping[self.enemy]) * 100.0, 2)
		self.printer.add(f"Best Enemy HP: {round(self.best_enemy_hp, 2)} / {self.enemy_hp_mapping[self.enemy]} ({enemy_hp_percent}% left) (Run #{self.best_enemy_hp_run})")
		self.printer.add("")
		self.printer.add("Latest Inputs:")
		self.printer.add("")

		for i in self.performed_inputs:
			self.printer.add(i)

		self.printer.flush()

		self.frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")

		action, label, game_input = self.ppo_agent.generate_action(self.frame_buffer)

		self.performed_inputs.appendleft(label)
		self.input_controller.handle_keys(game_input)


	def handle_fight_end(self, game_frame):
		reward = 0
		is_alive = False
		self.death_check = True
		self.fightstarted = False
		self.input_controller.handle_keys([])
		self.fightcount += 1
		self.handle_fight_training(game_frame)

	def reward_skullgirls(self, frames, **kwargs):
		reward = 0
		is_alive = self.health[0] + self.health[1]

		if is_alive:
			if self.health[0] < self.health[1]:
				self.multiplier_damage = 0
				return 0, True, False
			elif self.enemy_health[0] < self.enemy_health[1]:
				self.multiplier_damage += 0.05

				if self.multiplier_damage > 1:
					self.multiplier_damage = 1

				return (1 * self.multiplier_damage) + 0.001, True, False
			else:
				if self.enemy_health[0] < 20.0 and self._is_enemy_dead(frames[-2]):
					return 1, True, True

				return 0.001, True, False
		else:
			return 0, False, False

	def dump_metadata(self):
		metadata = dict(
			started_at=self.started_at,
			run_count=self.run_count - 1,
			observation_count=self.observation_count,
			reward_10=self.reward_10,
			reward_100=self.reward_100,
			reward_1000=self.reward_1000,
			rewards=self.rewards,
			average_reward_10=self.average_reward_10,
			average_reward_100=self.average_reward_100,
			average_reward_1000=self.average_reward_1000,
			top_reward=self.top_reward,
			top_reward_run=self.top_reward_run,
			enemy_hp_10=self.enemy_hp_10,
			enemy_hp_100=self.enemy_hp_100,
			enemy_hp_1000=self.enemy_hp_1000,
			average_enemy_hp_10=self.average_enemy_hp_10,
			average_enemy_hp_100=self.average_enemy_hp_100,
			average_enemy_hp_1000=self.average_enemy_hp_1000,
			best_enemy_hp=self.best_enemy_hp,
			best_enemy_hp_run=self.best_enemy_hp_run
		)

		with open("datasets/skullgirls/metadata.json", "wb") as f:
			f.write(pickle.dumps(metadata))

	def restore_metadata(self):
		with open("datasets/skullgirls/metadata.json", "rb") as f:
			metadata = pickle.loads(f.read())

		self.started_at = metadata["started_at"]
		self.run_count = metadata["run_count"]
		self.observation_count = metadata["observation_count"]
		self.reward_10 = metadata["reward_10"]
		self.reward_100 = metadata["reward_100"]
		self.reward_1000 = metadata["reward_1000"]
		self.rewards = metadata["rewards"]
		self.average_reward_10 = metadata["average_reward_10"]
		self.average_reward_100 = metadata["average_reward_100"]
		self.average_reward_1000 = metadata["average_reward_1000"]
		self.top_reward = metadata["top_reward"]
		self.top_reward_run = metadata["top_reward_run"]
		self.enemy_hp_10 = metadata["enemy_hp_10"]
		self.enemy_hp_100 = metadata["enemy_hp_100"]
		self.enemy_hp_1000 = metadata["enemy_hp_1000"]
		self.average_enemy_hp_10 = metadata["average_enemy_hp_10"]
		self.average_enemy_hp_100 = metadata["average_enemy_hp_100"]
		self.average_enemy_hp_1000 = metadata["average_enemy_hp_1000"]
		self.best_enemy_hp = metadata["best_enemy_hp"]
		self.best_enemy_hp_run = metadata["best_enemy_hp_run"]

	def handle_fight_training(self, game_frame):
		reward, is_alive, enemy_dead = self.reward_skullgirls([None, None, game_frame, None])
		self.printer.flush()
		self.printer.add("")
		self.printer.add("Updating Model With New Data... ")
		self.printer.flush()
		self.analytics_client.track(event_key="RUN_END", data=dict(run=self.run_count))
		reward = 0
		is_alive = False
		self.printer.flush()
		self.run_count += 1

		self.reward_10.appendleft(self.run_reward)
		self.reward_100.appendleft(self.run_reward)
		self.reward_1000.appendleft(self.run_reward)

		self.rewards.append(self.run_reward)

		self.average_reward_10 = float(np.mean(self.reward_10))
		self.average_reward_100 = float(np.mean(self.reward_100))
		self.average_reward_1000 = float(np.mean(self.reward_1000))

		if self.run_reward > self.top_reward:
			self.top_reward = self.run_reward
			self.top_reward_run = self.run_count - 1

			self.analytics_client.track(event_key="NEW_RECORD", data=dict(type="REWARD", value=self.run_reward, run=self.run_count - 1))

		self.analytics_client.track(event_key="EPISODE_REWARD", data=dict(reward=self.run_reward))

		self.previous_enemy_hp = 0 if enemy_dead else max(list(self.enemy_health)[:4])

		self.run_reward = 0

		self.enemy_hp_10.appendleft(self.previous_enemy_hp)
		self.enemy_hp_100.appendleft(self.previous_enemy_hp)
		self.enemy_hp_1000.appendleft(self.previous_enemy_hp)

		self.average_enemy_hp_10 = float(np.mean(self.enemy_hp_10))
		self.average_enemy_hp_100 = float(np.mean(self.enemy_hp_100))
		self.average_enemy_hp_1000 = float(np.mean(self.enemy_hp_1000))

		if (enemy_dead or self.previous_enemy_hp > 0) and self.previous_enemy_hp < self.best_enemy_hp:
			self.best_enemy_hp = self.previous_enemy_hp
			self.best_enemy_hp_run = self.run_count - 1

			self.analytics_client.track(event_key="NEW_RECORD", data=dict(type="BOSS_HP", value=self.previous_enemy_hp, run=self.run_count - 1))

		if not self.run_count % 10:
			self.ppo_agent.agent.save_model(directory=os.path.join(os.getcwd(), "datasets", "skullgirls", "ppo_model"), append_timestep=False)
			self.dump_metadata()

		self.health = collections.deque(np.full((16,), 6), maxlen=16)
		self.enemy_health = collections.deque(np.full((8,), self.enemy_hp_mapping[self.enemy]), maxlen=8)

		self.multiplier_damage = 0

		self.performed_inputs.clear()

		self.frame_buffer = None

		self.episode_started_at = time.time()
		self.episode_observation_count = 0
		time.sleep(1)
		self.take = 1
		self.handle_retry_button(game_frame)
