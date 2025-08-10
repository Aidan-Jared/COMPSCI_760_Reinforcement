
local socket = require "socket"

local data, msg_or_ip, port_or_nil

BalatrobotAPI = {}
BalatrobotAPI.socket = nil
BalatrobotAPI.clients = {}
BalatrobotAPI.last_state = nil
BalatrobotAPI.game_session_id = nil
BalatrobotAPI.game_start_time = nil
BalatrobotAPI.actions_enabled = true -- track player actions

function BalatrobotAPI.generate_session_id()
    return tostring(os.time()) .. "_" .. tostring(math.random(1000,9999))
end


function BalatrobotAPI.broadcast_gamestate()
    local _gamestate = Utils.getGamestate()

    -- session metadata
    _gamestate.session_id = BalatrobotAPI.game_session_id
    _gamestate.timestamp = os.time()
    _gamestate.game_time = BalatrobotAPI.game_start_time and (os.time() - BalatrobotAPI.game_start_time) or 0

    if G and G.STATE then
        _gamestate.current_state = G.STATE
        _gamestate.state_name = BalatrobotAPI.get_state_name(G.STATE)

        -- Add context-specific IDs
        if G.STATE == G.STATES.BLIND_SELECT and _gamestate.ante and _gamestate.ante.blinds then
            _gamestate.context_id = _gamestate.ante.blinds.selection_id
            _gamestate.context_type = "blind_selection"
        elseif G.STATE == G.STATES.SHOP and _gamestate.shop then
            _gamestate.context_id = _gamestate.shop.shop_id
            _gamestate.context_type = "shop_visit"
        end
    end

    if BalatrobotAPI.actions_enabled and ActionTracker then
        local blind_id = nil
        local shop_id = nil
        
        -- set new ids
        if _gamestate.ante and _gamestate.ante.blinds and _gamestate.ante.blinds.selection_id then
            blind_id = _gamestate.ante.blinds.selection_id
        end
        
        if _gamestate.shop and _gamestate.shop.shop_id then
            shop_id = _gamestate.shop.shop_id
        end
        
        -- pass to actions
        ActionTracker.set_last_context_ids(blind_id, shop_id)

        -- Clear context when moving to unrelated states
        if G and G.STATE then
            if G.STATE ~= G.STATES.BLIND_SELECT and G.STATE ~= G.STATES.SELECTING_HAND and G.STATE ~= G.STATES.HAND_PLAYED and G.STATE ~= G.STATES.DRAW_TO_HAND then
                ActionTracker.clear_blind_context()
            end
            if G.STATE ~= G.STATES.SHOP then
                ActionTracker.clear_shop_context()
            end
        end
        
        -- add recent actions 
        _gamestate.recent_actions = ActionTracker.get_all_actions()
    end

    -- add recent actions if action tracking enabled
    if BalatrobotAPI.actions_enabled and ActionTracker then
        _gamestate.recent_actions = ActionTracker.get_all_actions()
    end

    -- only send if state has change or in passive mode with send_all_states
    local state_change = not BalatrobotAPI.last_state or _gamestate.state_name ~= BalatrobotAPI.last_state.state_name
   
    if BALATRO_BOT_CONFIG.passive_mode and (BALATRO_BOT_CONFIG.send_all_states or state_change) then
        local _gamestateJsonString = json.encode(_gamestate)

        sendDebugMessage(_gamestateJsonString)

        --broadcast to all connected clients
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
    sendDebugMessage(tostring(json.encode(action)))
    local action_msg = {
        type = "action",
        action = action,
        timestamp = os.time(),
    }

    local action_json = json.encode(action_msg)

    for client_addr, client_port in pairs(BalatrobotAPI.clients) do
        if BalatrobotAPI.socket then
            BalatrobotAPI.socket:sendto(action_json, client_addr, client_port)
        end
    end
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
            -- register client
            BalatrobotAPI.clients[msg_or_ip] = port_or_nil
            BalatrobotAPI.respond("Connected to passive data stream", msg_or_ip, port_or_nil)
            sendDebugMessage('Client connected: ' .. msg_or_ip .. ':' .. port_or_nil)
            
            -- Send current state immediately
            BalatrobotAPI.broadcast_gamestate()
        elseif data == 'DISCONNECT\n' or data == 'DISCONNECT' then
            -- remove client
            BalatrobotAPI.clients[msg_or_ip] = nil
            BalatrobotAPI.respond("Disconnected", msg_or_ip, port_or_nil)
            sendDebugMessage('Client disconnected: ' .. msg_or_ip .. ':' .. port_or_nil)

        else
            BalatrobotAPI.respond("Passive mode - commands not accepted", msg_or_ip, port_or_nil)
        end
    elseif msg_or_ip ~= 'timeout' then
        sendDebugMessage("Network error: " .. tostring(msg_or_ip))
    end
    
    BalatrobotAPI.broadcast_gamestate()
end


function BalatrobotAPI.on_game_start()
    BalatrobotAPI.game_session_id = BalatrobotAPI.generate_session_id()
    BalatrobotAPI.game_start_time = os.time()

    if BalatrobotAPI.actions_enabled and ActionTracker then
        ActionTracker.init()
    end

    sendDebugMessage('New game session started: ' .. BalatrobotAPI.game_session_id)
    -- sendDebugMessage(G.SETTINGS.profile)
end

function BalatrobotAPI.on_game_end()
    if BalatrobotAPI.game_session_id then
        local final_state = Utils.getGamestate()
        final_state.session_id = BalatrobotAPI.game_session_id
        final_state.game_end = true
        final_state.final_timestamp = os.time()

        local final_json = json.encode(final_state)
        for client_addr, client_port in pairs(BalatrobotAPI.clients) do
            if BalatrobotAPI.socket then
                BalatrobotAPI.socket:sendto(final_json, client_addr, client_port)
            end
        end
        sendDebugMessage('Game session ended: ' .. BalatrobotAPI.game_session_id)
        BalatrobotAPI.game_session_id = nil
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
        -- ActionTracker.hook_consumable_actions()
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
