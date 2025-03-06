import traci
import logging

# Configure logging
logging.basicConfig(filename='traffic_simulation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def start_sumo():
    sumoCmd = ["sumo-gui", "-c", "osm.sumocfg"]
    traci.start(sumoCmd)

class AdaptiveTrafficLightAgent:
    def __init__(self, tls_id):
        self.tls_id = tls_id

    def adjust_phase_duration(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)

        new_duration = max(10, min(60, vehicle_count * 2))
        traci.trafficlight.setPhaseDuration(self.tls_id, new_duration)
        
        logging.info(f"Adaptive Agent - TLS {self.tls_id}: Vehicles: {vehicle_count}, New Duration: {new_duration}")

class RuleBasedTrafficLightAgent:
    def __init__(self, tls_id):
        self.tls_id = tls_id

    def switch_phase(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        vehicle_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)

        if vehicle_count > 10:
            traci.trafficlight.setPhase(self.tls_id, 2)  # Assuming phase 2 is green
        else:
            traci.trafficlight.setPhase(self.tls_id, 0)  # Assuming phase 0 is red
        
        logging.info(f"Rule-Based Agent - TLS {self.tls_id}: Vehicles: {vehicle_count}, Current phase: {traci.trafficlight.getPhase(self.tls_id)}")

def parse_tls_from_sumo():
    tls_list = []
    tls_ids = traci.trafficlight.getIDList()

    # Assign first half of traffic lights as adaptive, rest as rule-based
    for i, tls_id in enumerate(tls_ids):
        if i % 2 == 0:  # Alternate adaptive and rule-based
            tls_list.append(AdaptiveTrafficLightAgent(tls_id))
        else:
            tls_list.append(RuleBasedTrafficLightAgent(tls_id))

    return tls_list


def run_simulation(steps=5000):
    try:
        start_sumo()
        tls_agents = parse_tls_from_sumo()

        for step in range(steps):
            traci.simulationStep()
            for agent in tls_agents:
                if isinstance(agent, AdaptiveTrafficLightAgent):
                    agent.adjust_phase_duration()
                elif isinstance(agent, RuleBasedTrafficLightAgent):
                    agent.switch_phase()
            logging.info(f"Step {step}: Simulation running")
        logging.info(f"Simulation ended at step {steps}")
    except traci.exceptions.FatalTraCIError as e:
        logging.error(f"TraCI Error: {e}")
    finally:
        traci.close()
        logging.info("Simulation terminated gracefully")

if __name__ == "__main__":
    run_simulation()
