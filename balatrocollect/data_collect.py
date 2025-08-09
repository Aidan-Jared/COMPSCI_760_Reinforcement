#!/usr/bin/python3

import json
import socket
import time
import os
import signal
import sys
from datetime import datetime
from threading import Thread, Event

class BalatroDataCollector:
    def __init__(self, host = "localhost", port = 12345, save_dir="collected_data"):
        self.host = host
        self.port = port
        self.save_dir = save_dir
        self.sock = None
        self.running = False
        self.current_session = None
        self.session_data = []
        self.stop_event = Event()

        # make save dir
        os.makedirs(save_dir, exist_ok=True)

        # graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.stop()

    def connect(self):
        '''connect to BalatroAPI'''
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1.0) # time out

            self.sock.sendto(b"HELLO", (self.host, self.port))
            print(f"Connected to Balatro API at {self.host}:{self.port}")
            return True
        

        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        '''disconnect from API'''

        if self.sock:
            try:
                self.sock.sendto(b"DISCONNECT", (self.host, self.port))
            except:
                pass
            self.sock.close()
            self.sock = None

    def save_session_data(self):
        """Save collected session data to file"""
        if not self.session_data or not self.current_session:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self.current_session}_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        session_info = {
            "session_id": self.current_session,
            "collection_start": self.session_data[0].get("timestamp") if self.session_data else None,
            "collection_end": self.session_data[-1].get("timestamp") if self.session_data else None,
            "total_states": len(self.session_data),
            "states": self.session_data
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_info, f, indent=2)
            print(f"Saved session data to {filepath} ({len(self.session_data)} states)")
            
            # Also save as CSV for easy analysis
            self.save_session_csv(session_info, filepath.replace('.json', '.csv'))
            
        except Exception as e:
            print(f"Error saving session data: {e}")
    
    def save_session_csv(self, session_info, filepath):
        """Save session data as CSV for analysis"""
        try:
            import csv
            
            with open(filepath, 'w', newline='') as csvfile:
                if not session_info["states"]:
                    return
                    
                # Get all unique keys from all states
                all_keys = set()
                for state in session_info["states"]:
                    all_keys.update(state.keys())
                
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader()
                
                for state in session_info["states"]:
                    # Flatten nested objects for CSV
                    flattened = {}
                    for key, value in state.items():
                        if isinstance(value, (dict, list)):
                            flattened[key] = json.dumps(value)
                        else:
                            flattened[key] = value
                    writer.writerow(flattened)
                    
            print(f"Saved CSV data to {filepath}")
            
        except ImportError:
            print("CSV module not available, skipping CSV export")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def process_data(self, data):
        '''process received data, gamestate or action'''
        try:
            message = json.loads(data)

            # gets the session_id for refrerence and starts recording data
            session_id = message.get("session_id")

            if session_id and session_id != self.current_session:
                # save if new game
                if self.current_session and self.session_data:
                    self.save_session_data()

                self.current_session = session_id
                self.session_data = []
                print("session started: ")

            if message.get("type") == "action":
                # process player actions
                self.process_action(message["action"])
                return
            self.process_gamestate(message)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON received: {e}")
        except Exception as e:
            print(f"Error processing data: {e}")

    def process_action(self,action):
        '''process a players action in bot API format'''
        try:
            # Add to session data with special marking
            action_entry = {
                "type": "player_action", 
                "action_data": action,
                "received_timestamp": time.time()
            }

            if self.current_session:
                self.session_data.append(action_entry)
                action_type = action.get("action", "UNKNOWN")
                params = action.get("params", [])
                print("processed action: {}".format(action.get('state_name')))

        except Exception as e:
              print(f"Error processing action: {e}")
    
    def process_gamestate(self, gamestate):
        '''process received gamestate data'''
        try:
            # # see if new session
            # session_id = gamestate.get("session_id")
            # if session_id and session_id != self.current_session:
            #     if self.current_session and self.session_data:
            #         self.save_session_data()
            #     self.current_session = session_id
            #     self.session_data = []
            #     print("session started: ")
            
            # add timestamp if DNE
            if "received_timestamp" not in gamestate:
                gamestate["received_timestamp"] = time.time()
            
            # mark as gamestate
            gamestate["type"] = "game_state"

            # store game state
            self.session_data.append(gamestate)

            print("processed gamestate: {}".format(gamestate.get('state_name')))

            if gamestate.get("game_ended") or gamestate.get("state_name") == "GAME_OVER":
                print(f"Game ended for session {self.session_id}")
        
        except Exception as e:
            print(f"Error processing gamestate: {e}")
    
    def collect_data(self):
        '''main data collection loop'''
        self.running = True
        reconnect_delay = 5

        print("Starting Balatro data collection...")
        print("Play Balatro normally - data will be collected automatically")
        print("Press Ctrl+C to stop collection and save data")

        while self.running and not self.stop_event.is_set():
            if not self.sock:
                if not self.connect():
                    print(f"Retrying connection in {reconnect_delay} seconds...")
                    time.sleep(reconnect_delay)
                    continue
            try:
                data, addr = self.sock.recvfrom(6553600)
                self.process_data(data.decode('utf-8'))
            except socket.timeout:
                continue
            except socket.error as e:
                print(f"Socket error: {e}")
                self.disconnect()
                time.sleep(reconnect_delay)
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(1)
        
    def stop(self):
        """Stop data collection"""
        self.running = False
        self.stop_event.set()
        
        # Save any remaining session data
        if self.current_session and self.session_data:
            print("Saving final session data...")
            self.save_session_data()
        
        self.disconnect()
        print("Data collection stopped")

def main():

    import argparse

    parser = argparse.ArgumentParser(description="Collect Balatro gameplay data")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=12345, help="API port (default: 12346)")
    parser.add_argument("--save-dir", default="collected_data", help="Directory to save data (default: collected_data)")

    args = parser.parse_args()
    
    collector = BalatroDataCollector(args.host, args.port, args.save_dir)
    
    try:
        collector.collect_data()
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()

if __name__ == "__main__":
    main()