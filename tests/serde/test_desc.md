### input

```json
{
	"players": [
		{
			"name": "player 1",
			"state": "player state 1"
		},
		{
			"name": "player 2",
			"state": "player state 2"
		},
	],
	"characters": [
		{
			"name": "npc 1",
			"persona": {
				"short": "npc short 1",
				"long": "npc long 1"
			},
			"internal_state": "npc internal 1",
			"external_state": "npc external 1"
		},
		{
			"name": "npc 2",
			"persona": {
				"short": "npc short 2",
				"long": "npc long 2"
			},
			"internal_state": "npc internal 2",
			"external_state": "npc external 2"
		}
	],
	"world_events": {
		"past": [
			{
				"time": "past event time 1",
				"event": "past event event 1"
			},
			{
				"time": "past event time 2",
				"event": "past event event 2"
			}
		],
		"recent": [
			{
				"time": "recent event time 1",
				"event": "recent event event 1"
			},
			{
				"time": "recent event time 2",
				"event": "recent event event 2"
			}
		]
	},
	"messages": [
		{
			"role": "player",
			"name": "player 1",
			"content": "message 1"
		},
		{
			"role": "character",
			"name": "npc 1",
			"content": "message 2"
		},
		{
			"role": "character",
			"name": "npc 2",
			"content": "message 3"
		}
	]
}
```

# controller view 1

controller decides that npc 1 should speak next

```xml
<task_controller>

<PLAYERS>

<player_name> player 1
<state>
	player state 1
</state>

<player_name> player 2
<state>
	player state 2
</state>

<CHARACTERS>

<character_name> npc 1
<persona> 
	npc long 1
</persona>
<internal_state>
	npc internal 1
</internal_state>
<external_state>
	npc external 1
</external_state>

<character_name> npc 2
<persona> 
	npc long 2
</persona>
<internal_state>
	npc internal 2
</internal_state>
<external_state>
	npc external 2
</external_state>

<PAST_EVENTS>

<past_event> past event time 1
<event>
	past event event 1
</event>

<past_event> past event time 2
<event>
	past event event 2
</event>

<RECENT_EVENTS>

<recent_event> recent event time 1
<event>
    recent event event 1
</event>

<recent_event> recent event time 2
<event>
    recent event event 2
</event>

<MESSAGES>

<player>
	player 1
</player>
<text>
	message 1
</text>

<character>
	# target line
	npc 1
</character>
```

### controller view 2

controller decides that npc 2 should speak next

```xml
<task_controller>

<PLAYERS>

<player_name> player 1
<state>
	player state 1
</state>

<player_name> player 2
<state>
	player state 2
</state>

<CHARACTERS>

<character_name> npc 1
<persona> 
	npc long 1
</persona>
<internal_state>
	npc internal 1
</internal_state>
<external_state>
	npc external 1
</external_state>

<character_name> npc 2
<persona> 
	npc long 2
</persona>
<internal_state>
	npc internal 2
</internal_state>
<external_state>
	npc external 2
</external_state>

<PAST_EVENTS>

<past_event> past event time 1
<event>
	past event event 1
</event>

<past_event> past event time 2
<event>
	past event event 2
</event>

<RECENT_EVENTS>

<recent_event> recent event time 1
<event>
    recent event event 1
</event>

<recent_event> recent event time 2
<event>
    recent event event 2
</event>

<MESSAGES>

<player>
	player 1
</player>
<text>
	message 1
</text>

<character>
	npc 1
</character>
<text>
	message 2
</text>

<character>
	# target line
	npc 2
</character>
```

### controller view 3

controller decides that player should speak next

```xml
<task_controller>

<PLAYERS>

<player_name> player 1
<state>
	player state 1
</state>

<player_name> player 2
<state>
	player state 2
</state>

<CHARACTERS>

<character_name> npc 1
<persona> 
	npc long 1
</persona>
<internal_state>
	npc internal 1
</internal_state>
<external_state>
	npc external 1
</external_state>

<character_name> npc 2
<persona> 
	npc long 2
</persona>
<internal_state>
	npc internal 2
</internal_state>
<external_state>
	npc external 2
</external_state>

<PAST_EVENTS>

<past_event> past event time 1
<event>
	past event event 1
</event>

<past_event> past event time 2
<event>
	past event event 2
</event>

<RECENT_EVENTS>

<recent_event> recent event time 1
<event>
    recent event event 1
</event>

<recent_event> recent event time 2
<event>
    recent event event 2
</event>

<MESSAGES>

<player>
	player 1
</player>
<text>
	message 1
</text>

<character>
	npc 1
</character>
<text>
	message 2
</text>

<character>
	npc 1
</character>
<text>
	message 2
</text>

# target line
<player>
```

### model 1 view

```xml
<task_actor>

<PLAYERS>

<player_name> player 1
<state>
	player state 1
</state>

<player_name> player 2
<state>g
	player state 2
</state>

<CHARACTERS>

<character_name> npc 1
<persona> 
	npc long 1
</persona>
<internal_state>
	npc internal 1
</internal_state>
<external_state>
	npc external 1
</external_state>

<character_name> npc 2
<persona> 
	npc long 2
</persona>
<external_state>
	npc external 2
</external_state>

<PAST_EVENTS>

<past_event> past event time 1
<event>
	past event event 1
</event>

<past_event> past event time 2
<event>
	past event event 2
</event>

<RECENT_EVENTS>

<recent_event> recent event time 1
<event>
    recent event event 1
</event>

<recent_event> recent event time 2
<event>
    recent event event 2
</event>

<MESSAGES>

<player>
	player 1
</player>
<text>
	message 1
</text>

<character>
	npc 1
</character>
<persona>
	npc short 1
</persona>
<text>
	# target line
	message 2
</text>
```

### model 2 view

```xml
<task_actor>

<PLAYERS>

<player_name> player 1
<state>
	player state 1
</state>

<player_name> player 2
<state>
	player state 2
</state>

<CHARACTERS>

<character_name> npc 1
<persona> 
	npc long 1
</persona>
<external_state>
	npc external 1
</external_state>

<character_name> npc 2
<persona> 
	npc long 2
</persona>
<internal_state>
	npc internal 2
</internal_state>
<external_state>
	npc external 2
</external_state>

<PAST_EVENTS>

<past_event> past event time 1
<event>
	past event event 1
</event>

<past_event> past event time 2
<event>
	past event event 2
</event>

<RECENT_EVENTS>

<recent_event> recent event time 1
<event>
    recent event event 1
</event>

<recent_event> recent event time 2
<event>
    recent event event 2
</event>

<MESSAGES>

<player>
	player 1
</player>
<text>
	message 1
</text>

<character>
	npc 1
</character>
<text>
	message 2
</text>

<character>
	npc 2
</character>
<persona>
	npc short 2
</persona>
<text>
	# target line
	message 3
</text>
```
