## Json and Serialized Format for Chara

### Input and output json object

```json
{
	"players": [
		{
			"name": "jhuanglululu",
			"state": "information relevant to the current state of the player"
		},
		{
			"name": "Leon87_tw",
			"state": "entered the Big City 5 days ago, knows dark magic"
		},
	],
	"characters": [
		{
			"name": "Adventurer John",
			"persona": {
				"short": "short description that gets repeated everytime",
				"long": "long description that is only shown once at the very start"
			},
			"internal_state": "information relevant to the inner state of the character, ie. feelings, thoughts, secrets",
			"external_state": "information relevant to the current state of the character, ie. observable state"
		},
		{
			"name": "Adventurer Owen",
			"persona": {
				"short": "24 years old, male, brave, loves to drink",
				"long": "Owen is an adventurer from Hsinchu Kingdom. He is best known for his outdoor cooking skill..."
			},
			"internal_state": "Owen is happy because he bought a new spatula",
			"external_state": "Owen owned a cooking pan that can be used as a weapon."
		}
	],
	"world_events": {
		"past": [
			{
				"time": "time description",
				"event": "things that happened in the past(history)"
			},
			{
				"time": "March 11th, 2006",
				"event": "Adventurer Johna and Owen defeated the final boss inside the Great Dungeon"
			}
		],
		"recent": [
			{
				"time": "time description",
				"event": "things that happened recently and are more closely related to the conversation"
			},
			{
				"time": "April 8th, 2026",
				"event": "jhuanglululu and Leon87_tw went to the Great Dungeon"
			},
			{
				"time": "April 9th, 2026",
				"event": "Adventurer John and Adventurer Owen are going to bar"
			}
		]
	},
	"messages": [
		{
			"role": "player",
			"name": "jhuanglululu",
			"content": "Hello, we want to sell our loots, where can I find the adventure guild"
		},
		{
			"role": "character",
			"name": "Adventurer John",
			"content": "The guild is next to the bar right there"
		},
		{
			"role": "character",
			"name": "Adventurer Owen",
			"content": "We are heading to the bar, follow us"
		}
	]
}
```

---

### Serialized text

note:
- line started with # sign are comments, not actual serialzied text
- all formatting are just for readability, no new line and indentation will be included in serialized text unlessed explicitly inserted as part of a string

models & roles:

all the models are the same, but the context they can see are different

- model 0
    - decide who and when to speak
- model 1
    - role play adventurer 1
- model 2
    - role play adventurer 1

```xml
<PLAYERS>

<player_name> jhuanglululu
<state>
    information relevant to the current state of the player
</state>

<player_name> leon87_tw
<state>
    entered the Big City 5 days ago, knows dark magic
</state>

<CHARACTERS>

<character_name> Adventurer John
<persona> 
    long description that is only shown once at the very start
</persona>
# model 2 will not see John's internal state
<internal_state>
    information relevant to the inner state of the character, ie. feelings, thoughts
</internal_state>
<external_state>
    information relevant to the current state of the character
</external_state>

<character_name> Adventurer Owen
<persona>
    Owen is an adventurer from Hsinchu Kingdom. He is best known for his outdoor cooking skill...
</persona>
# model 1 will not see Owen's internal state
<internal_state>
    Owen is happy because he bought a new spatula
</internal_state>
<external_state>
    Owen owned a cooking pan that can be used as a weapon
</external_state>

<PAST_EVENTS>

<past_event> time description
<event>
    things that happened in the past(history)
</event>

<past_event> March 11th, 2006
<event>
    Adventurer Johna and Owen defeated the final boss inside the Great Dungeon
</event>

<RECENT_EVENTS>

<recent_event> time description
<event>
    things that happened recently and are more closely related to the conversation
</event>
<recent_event> April 8th, 2026
<event>
    jhuanglululu and Leon87_tw went to the Great Dungeon
</event>
<recent_event> April 9th, 2026
<event>
    Adventurer John and Adventurer Owen are going to bar
</event>

<MESSAGES>

<player>
	jhuanglululu
</player>
<text>
    Hello, we want to sell our look, where can I find the adventure guild
</text>

# model 0 decides that John(model 1) should speak
<character>
	Adventurer John
</character>
# model 1 gets the short persona of John
<persona>
    short description that gets repeated everytime
</persona>
<text>
    The guild is next to the bar right there
</text>
# short persona of John is dropped after text closed

# model 0 decides that Owen(model 2) should speak
<character>
	Adventurer Owen
</character>
# model 2 gets the short persona of Owen
<persona>
    24 years old, male, brave, loves to drink
</persona>
<text>
    We are heading to the bar, follow us
</text>
# short persona of Owen is dropped after text closed

# model 0 decides that player should speak
# when model 0 outputs <player>
# conversation is handoff to player
<player>
```

model 0 sample input

```xml
<PLAYERS>

<player_name> jhuanglululu
<state>
    information relevant to the current state of the player
</state>

<player_name> leon87_tw
<state>
    entered the Big City 5 days ago, knows dark magic
</state>

<CHARACTERS>

<character_name> Adventurer John
<persona> 
    long description that is only shown once at the very start
</persona>
<internal_state>
    information relevant to the inner state of the character, ie. feelings, thoughts
</internal_state>
<external_state>
    information relevant to the current state of the character
</external_state>

<character_name> Adventurer Owen
<persona>
    Owen is an adventurer from Hsinchu Kingdom. He is best known for his outdoor cooking skill...
</persona>
<internal_state>
    Owen is happy because he bought a new spatula
</internal_state>
<external_state>
    Owen owned a cooking pan that can be used as a weapon
</external_state>

<PAST_EVENTS>

<past_event> time description
<event>
    things that happened in the past(history)
</event>

<past_event> March 11th, 2006
<event>
    Adventurer Johna and Owen defeated the final boss inside the Great Dungeon
</event>

<RECENT_EVENTS>

<recent_event> time description
<event>
    things that happened recently and are more closely related to the conversation
</event>
<recent_event> April 8th, 2026
<event>
    jhuanglululu and Leon87_tw went to the Great Dungeon
</event>
<recent_event> April 9th, 2026
<event>
    Adventurer John and Adventurer Owen are going to bar
</event>

<MESSAGES>

<player>
	jhuanglululu
</player>
<text>
    Hello, we want to sell our look, where can I find the adventure guild
</text>

<character>
# output starts here
```

---

model 1 sample input

```xml
<PLAYERS>

<player_name> jhuanglululu
<state>
    information relevant to the current state of the player
</state>

<player_name> leon87_tw
<state>
    entered the Big City 5 days ago, knows dark magic
</state>

<CHARACTERS>

<character_name> Adventurer John
<persona> 
    long description that is only shown once at the very start
</persona>
<internal_state>
    information relevant to the inner state of the character, ie. feelings, thoughts
</internal_state>
<external_state>
    information relevant to the current state of the character
</external_state>

<character_name> Adventurer Owen
<persona>
    Owen is an adventurer from Hsinchu Kingdom. He is best known for his outdoor cooking skill...
</persona>
<external_state>
    Owen owned a cooking pan that can be used as a weapon
</external_state>

<PAST_EVENTS>

<past_event> time description
<event>
    things that happened in the past(history)
</event>

<past_event> March 11th, 2006
<event>
    Adventurer Johna and Owen defeated the final boss inside the Great Dungeon
</event>

<RECENT_EVENTS>

<recent_event> time description
<event>
    things that happened recently and are more closely related to the conversation
</event>
<recent_event> April 8th, 2026
<event>
    jhuanglululu and Leon87_tw went to the Great Dungeon
</event>
<recent_event> April 9th, 2026
<event>
    Adventurer John and Adventurer Owen are going to bar
</event>

<MESSAGES>

<player>
	jhuanglululu
</player>
<text>
    Hello, we want to sell our look, where can I find the adventure guild
</text>

<character>
	Adventurer John
</character>
<persona>
    short description that gets repeated everytime
</persona>
<text>
# output starts here
```

---

### Token Estimation

estimated with [GPT5.X](https://platform.openai.com/tokenizer)

| Section                  | Token Count |
|--------------------------|-------------|
| full context             | ~600        |
| full description         | ~400        |
| player                   | ~50         |
| characters               | ~150        |
| events                   | ~200        |
| full message             | ~200        |
| single turn with persona | ~50         |
| single text section      | ~20         |

Estimation

| Conversation Length | Message Count | Token Count |
|---------------------|---------------|-------------|
| short               | 10            | ~1000       |
| medium              | 25            | ~1500       |
| long                | 50            | ~2500       |

Earlier models: 1k
Final model: 4k or 8k
