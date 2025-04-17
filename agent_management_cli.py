from vertexai import agent_engines

def list_agents():
    print("📋 Available Agents:")
    print("---------------")
    agents = agent_engines.list()
    agent_map = {}

    for idx, agent in enumerate(agents, start=1):
        gca = getattr(agent, "gca_resource", None)
        if gca:
            agent_map[str(idx)] = gca
            print(f"[{idx}] Display Name: {gca.display_name}")
            print(f"     Resource ID: {gca.name.split('/')[-1]}")
            print(f"     Description: {gca.description}")
            print(f"     Created: {gca.create_time}")
            print(f"     Updated: {gca.update_time}")
            print("---------------")
    return agent_map

def prompt_action():
    while True:
        action = input("What would you like to do? [D]elete / [U]pdate / [C]ancel: ").strip().lower()
        if action in ['d', 'u', 'c']:
            return action
        print("❌ Invalid option. Please enter D, U, or C.")

def prompt_agent_choice(agent_map):
    while True:
        choice = input("Enter the number of the agent you'd like to act on: ").strip()
        if choice in agent_map:
            return agent_map[choice]
        print("❌ Invalid agent number.")

def confirm(prompt="Are you sure? (y/n): "):
    return input(prompt).strip().lower() == 'y'

def delete_agent(agent):
    print(f"⚠️ You are about to delete agent: {agent.display_name}")
    print("⚠️ This will also delete all child sessions and memories if force-delete is enabled.")

    if not confirm("Type 'y' to confirm deletion: "):
        print("❌ Deletion canceled.")
        return

    use_force = confirm("Also delete child resources? Type 'y' to enable force delete: ")

    try:
        agent_engines.delete(agent.name, force=use_force)
        print("🗑️ Agent deleted successfully.")
    except Exception as e:
        print(f"❌ Failed to delete agent: {e}")


def update_agent(agent):
    print(f"✏️ You are updating agent: {agent.display_name}")
    new_name = input(f"New Display Name (press Enter to keep '{agent.display_name}'): ").strip()
    new_desc = input(f"New Description (press Enter to keep existing): ").strip()

    updates = {}
    if new_name:
        updates["display_name"] = new_name
    if new_desc:
        updates["description"] = new_desc

    if not updates:
        print("ℹ️ No changes provided. Skipping update.")
        return

    print("⬆️ Updates to be applied:")
    for k, v in updates.items():
        print(f"  {k}: {v}")

    if confirm("Type 'y' to confirm update: "):
        agent_engines.update(
            resource_name=agent.name,
            display_name=updates.get("display_name"),
            description=updates.get("description")
        )
        print("✅ Agent updated.")
    else:
        print("❌ Update canceled.")

def main():
    print("🔧 Vertex AI Agent Management CLI")
    print("===============================")

    while True:
        agent_map = list_agents()
        if not agent_map:
            print("🚫 No agents found.")
            break

        action = prompt_action()
        if action == 'c':
            print("👋 Operation canceled.")
            break

        agent = prompt_agent_choice(agent_map)
        if action == 'd':
            delete_agent(agent)
        elif action == 'u':
            update_agent(agent)

        again = input("🔄 Would you like to manage another agent? (y/n): ").strip().lower()
        if again != 'y':
            print("✅ Exiting Agent Management CLI.")
            break


if __name__ == "__main__":
    main()
