
local socket = require "socket"

local data, msg_or_ip, port_or_nil

BalatrobotAPI = {}
BalatrobotAPI.socket = nil
BalatrobotAPI.clients = {}
BalatrobotAPI.last_state = nil
BalatrobotAPI.game_start_time = nil
BalatrobotAPI.actions_enabled = true -- track player actions

BalatrobotAPI.state_frame_counters = {} -- Track frames since state entered
BalatrobotAPI.delayed_states = {"SHOP", "TAROT_PACK", "PLANET_PACK", "SPECTRAL_PACK", "STANDARD_PACK", "BUFFOON_PACK"} -- States that need delay
BalatrobotAPI.delay_frames = 120 -- Number of frames to wait
BalatrobotAPI.delayed_broadcasts_sent = {}
BalatrobotAPI.current_tracked_state = nil -- Track current state for frame counting
BalatrobotAPI.first_entry_per_round = {}
BalatrobotAPI.current_round_key = nil

function BalatrobotAPI.broadcast_gamestate()
    local _gamestate = Utils.getGamestate()

    -- Add API-specific metadata
    _gamestate.timestamp = os.time()
    _gamestate.game_time = BalatrobotAPI.game_start_time and (os.time() - BalatrobotAPI.game_start_time) or 0

    if G and G.STATE then
        _gamestate.current_state = G.STATE
        _gamestate.state_name = BalatrobotAPI.get_state_name(G.STATE)
        if _gamestate.state_name == "DRAW_TO_HAND" then
            return;
        end
    end

    if not Utils.current_gamestate_id then
        Utils.current_gamestate_id = Utils.generateGamestateId()
    end
    _gamestate.gamestate_id = Utils.current_gamestate_id
    local state_changed = BalatrobotAPI.current_tracked_state ~= _gamestate.gamestate_id

    -- NEW: Track round changes to reset first_entry tracking
    local round_key = tostring(G.GAME.round) .. "_" .. tostring(G.GAME.round_resets.ante)
    if BalatrobotAPI.current_round_key ~= round_key then
        BalatrobotAPI.current_round_key = round_key
        BalatrobotAPI.first_entry_per_round = {} -- Reset first entry tracking for new round
        sendDebugMessage("New round detected (" .. round_key .. ") - reset first entry tracking")
    end

    -- Initialize or update frame counter
    if state_changed then
        BalatrobotAPI.current_tracked_state = _gamestate.gamestate_id
        BalatrobotAPI.state_frame_counters[_gamestate.state_name] = 0
        BalatrobotAPI.delayed_broadcasts_sent[_gamestate.state_name] = false
        sendDebugMessage("State changed to: " .. _gamestate.state_name .. " - resetting frame counter.")
    else
        local current_count = BalatrobotAPI.state_frame_counters[_gamestate.state_name] or 0
        BalatrobotAPI.state_frame_counters[_gamestate.state_name] = current_count + 1
    end

    local current_frame_count = BalatrobotAPI.state_frame_counters[_gamestate.state_name]
    local should_send = false

    if BALATRO_BOT_CONFIG.passive_mode then
        if BalatrobotAPI.needs_frame_delay(_gamestate.state_name) then
            -- NEW: Check if this is the first entry to this state in current round
            local is_first_entry = not BalatrobotAPI.first_entry_per_round[_gamestate.state_name]
            
            if is_first_entry then
                -- First entry - apply delay as before
                if BALATRO_BOT_CONFIG.send_all_states then
                    if current_frame_count >= BalatrobotAPI.delay_frames then
                        should_send = true
                        BalatrobotAPI.first_entry_per_round[_gamestate.state_name] = true
                    end
                else
                    -- Send only once after the delay period
                    if G.STATE == G.STATES.TAROT_PACK or G.STATE == G.STATES.SPECTRAL_PACK then
                        if current_frame_count >= BalatrobotAPI.delay_frames * 5 and 
                       not BalatrobotAPI.delayed_broadcasts_sent[_gamestate.state_name] then
                        should_send = true
                        BalatrobotAPI.first_entry_per_round[_gamestate.state_name] = true
                        end
                    else
                        if current_frame_count >= BalatrobotAPI.delay_frames and 
                       not BalatrobotAPI.delayed_broadcasts_sent[_gamestate.state_name] then
                        should_send = true
                        BalatrobotAPI.first_entry_per_round[_gamestate.state_name] = true
                        end
                    end
                end
                
                if should_send then
                    sendDebugMessage("First entry to " .. _gamestate.state_name .. " - sent after delay of " .. current_frame_count .. " frames")
                end
            else
                -- Subsequent entry in same round - no delay, send immediately
                if BALATRO_BOT_CONFIG.send_all_states or state_changed then
                    should_send = true
                    sendDebugMessage("Subsequent entry to " .. _gamestate.state_name .. " in same round - sent immediately")
                end
            end
        else
            -- No delay needed for this state
            if BALATRO_BOT_CONFIG.send_all_states or state_changed then
                should_send = true
            end
        end
    end
    
    -- ROBUSTNESS CHECK: For pack states, ensure the card data is populated before sending.
    -- This check will override the 'should_send' flag if the data is not ready.
    if should_send and Utils.isPackState(G.STATE) then
        local hand_data_is_read = nil
        local pack_data_is_ready = G.pack_cards and G.pack_cards.cards and #G.pack_cards.cards > 0
        if G.STATE == G.STATES.TAROT_PACK or G.STATE == G.STATES.SPECTRAL_PACK then
            hand_data_is_read = G.hand and G.hand.cards and #G.hand.cards > 7
            if not pack_data_is_ready and not hand_data_is_read  then
                should_send = false -- Defer the broadcast
                sendDebugMessage("Pack state (" .. _gamestate.state_name .. ") detected, but pack data is not ready. Deferring broadcast.")
                Utils.pack_size = nil
            else
                sendDebugMessage("Pack data is ready for broadcast with " .. #G.pack_cards.cards .. " cards.")
                Utils.cache_pack_positions()
            end
        else
            if not pack_data_is_ready  then
                should_send = false -- Defer the broadcast
                sendDebugMessage("Pack state (" .. _gamestate.state_name .. ") detected, but pack data is not ready. Deferring broadcast.")
                Utils.pack_size = nil
            else
                sendDebugMessage("Pack data is ready for broadcast with " .. #G.pack_cards.cards .. " cards.")
                Utils.cache_pack_positions()
            end
        end
    end

    if should_send then
        -- If this is a delayed broadcast being sent for the first time, set the flag now
        if BalatrobotAPI.needs_frame_delay(_gamestate.state_name) and not BALATRO_BOT_CONFIG.send_all_states then
             BalatrobotAPI.delayed_broadcasts_sent[_gamestate.state_name] = true
             sendDebugMessage("Setting 'sent' flag for delayed broadcast of " .. _gamestate.state_name)
        end
        -- save card locations 
        Utils.cache_booster_positions()
        Utils.cache_shop_card_positions()

        -- get gamestate id       
        local _gamestateJsonString = json.encode(_gamestate)

        sendDebugMessage("Broadcasting gamestate: " .. _gamestate.state_name .. 
                         " (session: " .. tostring(_gamestate.session_id) .. 
                         ", context: " .. tostring(_gamestate.context_id) .. ")")

        -- Broadcast to all connected clients
        for client_addr, client_port in pairs(BalatrobotAPI.clients) do
            if BalatrobotAPI.socket then
                BalatrobotAPI.socket:sendto(_gamestateJsonString, client_addr, client_port)
            end
        end

        BalatrobotAPI.last_state = _gamestate
    end
end
 

function BalatrobotAPI.broadcast_action(action)
    -- send out individual action immediatly
    local action_msg = {
        type = "action",
        action = action,
        timestamp = os.time(),
    }

    local action_json = json.encode(action_msg)

    sendDebugMessage('\n')
    sendDebugMessage("Broadcasting action: " .. action.action .. 
                    " (context: " .. tostring(action.context_id) .. ")")
    sendDebugMessage('\n')

    for client_addr, client_port in pairs(BalatrobotAPI.clients) do
        if BalatrobotAPI.socket then
            BalatrobotAPI.socket:sendto(action_json, client_addr, client_port)
        end
    end
end

-- Helper function to check if a state needs delay
function BalatrobotAPI.needs_frame_delay(state_name)
    for _, delayed_state in ipairs(BalatrobotAPI.delayed_states) do
        if state_name == delayed_state then
            return true
        end
    end
    return false
end

function BalatrobotAPI.get_state_name(state)
    local state_names = {
        [G.STATES.MENU] = "MENU",
        [G.STATES.SELECTING_HAND] = "SELECTING_HAND", 
        [G.STATES.HAND_PLAYED] = "HAND_PLAYED",
        [G.STATES.DRAW_TO_HAND] = "DRAW_TO_HAND",
        [G.STATES.GAME_OVER] = "GAME_OVER",
        [G.STATES.SHOP] = "SHOP",
        [G.STATES.BLIND_SELECT] = "BLIND_SELECT",
        [G.STATES.ROUND_EVAL] = "ROUND_EVAL",
        [G.STATES.TAROT_PACK] = "TAROT_PACK",
        [G.STATES.PLANET_PACK] = "PLANET_PACK",
        [G.STATES.SPECTRAL_PACK] = "SPECTRAL_PACK",
        [G.STATES.STANDARD_PACK] = "STANDARD_PACK",
        [G.STATES.BUFFOON_PACK] = "BUFFOON_PACK",
        [G.STATES.NEW_ROUND] = "NEW_ROUND"
    }
    return state_names[state] or "UNKNOWN"
end

function BalatrobotAPI.respond(str, client_addr, client_port)
    if BalatrobotAPI.socket and client_addr and client_port then
        local response = {
            response = str,
            timestamp = os.time()
        }
        local response_str = json.encode(response)
        BalatrobotAPI.socket:sendto(response_str, client_addr, client_port)
    end
end


function BalatrobotAPI.update(dt)
    if not BalatrobotAPI.socket then
        BalatrobotAPI.socket = socket.udp()
        BalatrobotAPI.socket:settimeout(0)
        local port = arg[1] or BALATRO_BOT_CONFIG.port
        BalatrobotAPI.socket:setsockname('127.0.0.1', tonumber(port))
        sendDebugMessage('Passive API socket created on port ' .. port)
    end

    local data, msg_or_ip, port_or_nil = BalatrobotAPI.socket:receivefrom()
    if data then
        if data == 'HELLO\n' or data == 'HELLO' then
            BalatrobotAPI.clients[msg_or_ip] = port_or_nil
            BalatrobotAPI.respond("Connected to passive data stream", msg_or_ip, port_or_nil)
            sendDebugMessage('Client connected: ' .. msg_or_ip .. ':' .. port_or_nil)
            
            -- Send current state immediately (will respect delay logic)
            BalatrobotAPI.broadcast_gamestate()
        elseif data == 'DISCONNECT\n' or data == 'DISCONNECT' then
            BalatrobotAPI.clients[msg_or_ip] = nil
            BalatrobotAPI.respond("Disconnected", msg_or_ip, port_or_nil)
            sendDebugMessage('Client disconnected: ' .. msg_or_ip .. ':' .. port_or_nil)
        else
            BalatrobotAPI.respond("Passive mode - commands not accepted", msg_or_ip, port_or_nil)
        end
    elseif msg_or_ip ~= 'timeout' then
        sendDebugMessage("Network error: " .. tostring(msg_or_ip))
    end
    
    -- Always call broadcast_gamestate - it handles the delay logic internally
    BalatrobotAPI.broadcast_gamestate()
end

-- Clean up frame counters on game end
function BalatrobotAPI.on_game_end()
    if Utils.current_session_id then
        local final_state = Utils.getGamestate()
        final_state.game_end = true
        final_state.final_timestamp = os.time()

        local final_json = json.encode(final_state)
        for client_addr, client_port in pairs(BalatrobotAPI.clients) do
            if BalatrobotAPI.socket then
                BalatrobotAPI.socket:sendto(final_json, client_addr, client_port)
            end
        end
        sendDebugMessage('Game session ended: ' .. Utils.current_session_id)
        
        -- Clear frame counters and first entry tracking
        BalatrobotAPI.state_frame_counters = {}
        BalatrobotAPI.delayed_broadcasts_sent = {}
        BalatrobotAPI.first_entry_per_round = {} -- NEW: Clear first entry tracking
        BalatrobotAPI.current_tracked_state = nil
        BalatrobotAPI.current_round_key = nil -- NEW: Clear round tracking
        BalatrobotAPI.game_start_time = nil
    end
end


-- Clean up frame counters on game end
function BalatrobotAPI.on_game_end()
    if Utils.current_session_id then
        local final_state = Utils.getGamestate()
        final_state.game_end = true
        final_state.final_timestamp = os.time()

        local final_json = json.encode(final_state)
        for client_addr, client_port in pairs(BalatrobotAPI.clients) do
            if BalatrobotAPI.socket then
                BalatrobotAPI.socket:sendto(final_json, client_addr, client_port)
            end
        end
        sendDebugMessage('Game session ended: ' .. Utils.current_session_id)
        
        -- Clear frame counters and first entry tracking
        BalatrobotAPI.state_frame_counters = {}
        BalatrobotAPI.delayed_broadcasts_sent = {}
        BalatrobotAPI.first_entry_per_round = {} -- NEW: Clear first entry tracking
        BalatrobotAPI.current_tracked_state = nil
        BalatrobotAPI.current_round_key = nil -- NEW: Clear round tracking
        BalatrobotAPI.game_start_time = nil
    end
end


function BalatrobotAPI.init()
    love.update = Hook.addcallback(love.update, BalatrobotAPI.update)

    -- Initialize action tracking if enabled
    if BalatrobotAPI.actions_enabled then
        ActionTracker.hook_hand_actions()
        ActionTracker.hook_blind_actions()
        ActionTracker.hook_shop_actions()
        ActionTracker.hook_booster_actions()
        ActionTracker.hook_selling_actions()
        ActionTracker.hook_rearrange_actions()
        ActionTracker.hook_run_start()
    end

    -- Tell the game engine that every frame is 8/60 seconds long
    -- Speeds up the game execution
    -- Values higher than this seem to cause instability
    if BALATRO_BOT_CONFIG.dt then
        love.update = Hook.addbreakpoint(love.update, function(dt)
            return BALATRO_BOT_CONFIG.dt
        end)
    end

    -- Disable FPS cap
    if BALATRO_BOT_CONFIG.uncap_fps then
        G.FPS_CAP = 999999.0
    end

    -- Makes things move instantly instead of sliding
    if BALATRO_BOT_CONFIG.instant_move then
        function Moveable.move_xy(self, dt)
            -- Directly set the visible transform to the target transform
            self.VT.x = self.T.x
            self.VT.y = self.T.y
        end
    end

    -- Forcibly disable vsync
    if BALATRO_BOT_CONFIG.disable_vsync then
        love.window.setVSync(0)
    end

    -- Disable card scoring animation text
    if BALATRO_BOT_CONFIG.disable_card_eval_status_text then
        card_eval_status_text = function(card, eval_type, amt, percent, dir, extra) end
    end

    -- Only draw/present every Nth frame
    local original_draw = love.draw
    local draw_count = 0
    love.draw = function()
        draw_count = draw_count + 1
        if draw_count % BALATRO_BOT_CONFIG.frame_ratio == 0 then
            original_draw()
        end
    end

    -- Hook game start/end events
    if G and G.FUNCS then
        G.FUNCS.start_run = Hook.addcallback(G.FUNCS.start_run, BalatrobotAPI.on_game_start)
    end

    --hook state changes to detect game over
    local original_set_state = G.STATE_MANAGER and G.STATE_MANAGER.set_state
    if original_set_state then
        G.STATE_MANAGER.set_state = function (self, state, ...)
            if state == G.STATES.GAME_OVER and G.STATE ~= G.STATES.GAME_OVER then
                BalatrobotAPI.on_game_end()
            end
            return original_set_state(self, state, ...)
        end
    end

    sendDebugMessage('Passive API initialized with processed action tracking')
end

return BalatrobotAPI