ActionTracker = {}
ActionTracker.actions = {}

function ActionTracker.init()
    ActionTracker.actions = {}
    sendDebugMessage("ActionTracker initialized for session: " .. tostring(Utils.current_session_id))
end


function ActionTracker.log_action(action_type, params)
    
    Utils.ensureSessionId()

    local current_context_ids = Utils.getCurrentContextIds()
    
    local action = {
        timestamp = os.time(),
        game_time = BalatrobotAPI.game_start_time and (os.time() - BalatrobotAPI.game_start_time) or 0,
        session_id = current_context_ids.session_id,
        action = action_type,
        params = params or {},
        game_state = G.STATE and BalatrobotAPI.get_state_name(G.STATE) or "UNKNOWN",
    }

    -- add all avaliable context ids
    if current_context_ids.round_id then
        action.round_id = current_context_ids.round_id
    end

    -- Use the most relevant context ID based on current state
    if G and G.STATE then
        if Utils.isBlindContextState(G.STATE) and current_context_ids.blind_id then
            action.context_id = current_context_ids.blind_id
            action.context_type = "blind_session"
            action.blind_id = current_context_ids.blind_id
        elseif Utils.isShopContextState(G.STATE) and current_context_ids.shop_id then
            action.context_id = current_context_ids.shop_id
            action.context_type = "shop_session"  
            action.shop_id = current_context_ids.shop_id
        -- Fallback to any available context ID
        elseif current_context_ids.shop_id then
            action.context_id = current_context_ids.shop_id
            action.context_type = "shop_session"
            action.shop_id = current_context_ids.shop_id
        elseif current_context_ids.blind_id then
            action.context_id = current_context_ids.blind_id
            action.context_type = "blind_session"
            action.blind_id = current_context_ids.blind_id
        end
    end

    table.insert(ActionTracker.actions, action)
    sendDebugMessage("Action logged: " .. action_type .. " with context: " .. tostring(action.context_id or "none"))

    BalatrobotAPI.broadcast_action(action)
end

-- Clear actions when starting new context sessions
function ActionTracker.clear_actions_for_new_context(context_type)
    sendDebugMessage("Clearing actions for new " .. context_type .. " context")
    ActionTracker.actions = {}
end


-- track finished hand decisioins
function ActionTracker.hook_hand_actions()
    -- Play button
    if G.FUNCS.play_cards_from_highlighted then
        G.FUNCS.play_cards_from_highlighted = Hook.addcallback(G.FUNCS.play_cards_from_highlighted, function (e)
            local card_positons = {}
            if G.hand then
                for i, card in ipairs(G.hand.cards) do
                    if card.highlighted then
                        table.insert(card_positons,i)
                    end
                end
            end

            if #card_positons > 0 then
                ActionTracker.log_action("PLAY_HAND", card_positons)
            end
        end)
    end

    if G.FUNCS.discard_cards_from_highlighted then
        G.FUNCS.discard_cards_from_highlighted = Hook.addcallback(G.FUNCS.discard_cards_from_highlighted, function (e)
            local card_positons = {}
            if G.hand then
                for i, card in ipairs(G.hand.cards) do
                    if card.highlighted then
                        table.insert(card_positons, i)
                    end
                end
            end
            if #card_positons > 0 then
                ActionTracker.log_action("DISCARD_HAND", card_positons)
            end
        end)
    end
end


-- Track blind selection decisioins
function ActionTracker.hook_blind_actions()
    -- Select blind
    if G.FUNCS.select_blind then
        G.FUNCS.select_blind = Hook.addcallback(G.FUNCS.select_blind, function(e)
            ActionTracker.log_action("SELECT_BLIND", {})
        end)
    end
    
    -- Skip blind
    if G.FUNCS.skip_blind then
        G.FUNCS.skip_blind = Hook.addcallback(G.FUNCS.skip_blind, function(e)
            ActionTracker.log_action("SKIP_BLIND", {})
        end)
    end
end

-- Track shop decisions
-- function ActionTracker.hook_shop_actions()
--     -- Shop reroll
--     if G.FUNCS.reroll_shop then
--         G.FUNCS.reroll_shop = Hook.addcallback(G.FUNCS.reroll_shop, function(e)
--             ActionTracker.log_action("REROLL_SHOP", {})
--         end)
--     end
    
--     -- End shop
--     if G.FUNCS.toggle_shop then
--         G.FUNCS.toggle_shop = Hook.addcallback(G.FUNCS.toggle_shop, function(e)
--             ActionTracker.log_action("END_SHOP", {})
--         end)
--     end
    
--     -- Buy joker/card from shop
--     if G.FUNCS.buy_from_shop then
--         G.FUNCS.buy_from_shop = Hook.addcallback(G.FUNCS.buy_from_shop, function(e)
--             local card = e and e.card
--             sendDebugMessage('bought joker ' .. tostring(json.encode(G.FUNCS.buy_from_shop)))
--             if card and G.shop_jokers and G.shop_jokers.cards then
--                 -- Find position in shop
--                 for i, shop_card in ipairs(G.shop_jokers.cards) do
--                     if shop_card == card then
--                         ActionTracker.log_action("BUY_CARD", {i})
--                         break
--                     end
--                 end
--             end
--         end)
--     end
    
--     -- Buy voucher
--     if G.FUNCS.buy_voucher then
--         G.FUNCS.buy_voucher = Hook.addcallback(G.FUNCS.buy_voucher, function(e)
--             local card = e and e.card
--             if card and G.shop_vouchers and G.shop_vouchers.cards then
--                 -- Find position in shop vouchers
--                 for i, voucher_card in ipairs(G.shop_vouchers.cards) do
--                     if voucher_card == card then
--                         ActionTracker.log_action("BUY_VOUCHER", {i})
--                         break
--                     end
--                 end
--             end
--         end)
--     end
    
--     -- Buy booster pack
--     if G.FUNCS.buy_and_use_consumeable then
--         G.FUNCS.buy_and_use_consumeable = Hook.addcallback(G.FUNCS.buy_and_use_consumeable, function(e)
--             local card = e and e.card
--             if card and G.shop_booster and G.shop_booster.cards then
--                 -- Find position in shop boosters
--                 for i, booster_card in ipairs(G.shop_booster.cards) do
--                     if booster_card == card then
--                         ActionTracker.log_action("BUY_BOOSTER", {i})
--                         break
--                     end
--                 end
--             end
--         end)
--     end
-- end

function ActionTracker.hook_shop_actions()
    -- Shop reroll
    if G.FUNCS.reroll_shop then
        G.FUNCS.reroll_shop = Hook.addcallback(G.FUNCS.reroll_shop, function(e)
            ActionTracker.log_action("REROLL_SHOP", {})
        end)
    end
    
    -- End shop
    if G.FUNCS.toggle_shop then
        G.FUNCS.toggle_shop = Hook.addcallback(G.FUNCS.toggle_shop, function(e)
            ActionTracker.log_action("END_SHOP", {})
        end)
    end
    
    -- Buy joker/card from shop
    if G.FUNCS.buy_from_shop then
        G.FUNCS.buy_from_shop = Hook.addcallback(G.FUNCS.buy_from_shop, function(e)
            -- Get the card from e.config.ref_table (as shown in the original function)
            local card = e.config.ref_table
        
            if card and card:is(Card) then
                local card_data = {
                    key = card.config.center.key,
                    name = card.config and card.config.center and card.config.center.name,
                    set = card.ability and card.ability.set,
                    cost = card.cost,
                    edition = card.edition,
                    seal = card.seal
                }
                
                -- Find position in appropriate shop area
                local position = nil
                
                -- Check jokers
                if G.shop_jokers and G.shop_jokers.cards then
                    for i, shop_card in ipairs(G.shop_jokers.cards) do
                        if shop_card == card then
                            position = i
                            card_data.shop_area = "jokers"
                            break
                        end
                    end
                end
                
                -- Check consumables if not found in jokers
                if not position and G.shop_consumeables and G.shop_consumeables.cards then
                    for i, shop_card in ipairs(G.shop_consumeables.cards) do
                        if shop_card == card then
                            position = i
                            card_data.shop_area = "consumeables"
                            break
                        end
                    end
                end
                
                -- Check playing cards if not found elsewhere
                if not position and G.shop_deck and G.shop_deck.cards then
                    for i, shop_card in ipairs(G.shop_deck.cards) do
                        if shop_card == card then
                            position = i
                            card_data.shop_area = "deck"
                            break
                        end
                    end
                end
                
                if position then
                    card_data.position = position
                    ActionTracker.log_action("BUY_CARD", card_data)
                else
                    -- Log anyway with available data
                    ActionTracker.log_action("BUY_CARD", card_data)
                end
                
                sendDebugMessage('Bought card: ' .. tostring(card_data.name or card_data.key or "unknown"))
            else
                sendDebugMessage('Could not get card data from buy_from_shop event')
            end
        end)
    end
    
    -- Buy voucher
    if G.FUNCS.buy_voucher then
        G.FUNCS.buy_voucher = Hook.addcallback(G.FUNCS.buy_voucher, function(e)
            local card = e.config.ref_table
            
            if card and card:is(Card) then
                local card_data = {
                    key = card.config.center.key,
                    name = card.config and card.config.center and card.config.center.name,
                    cost = card.cost
                }
                
                -- Find position in shop vouchers
                if G.shop_vouchers and G.shop_vouchers.cards then
                    for i, voucher_card in ipairs(G.shop_vouchers.cards) do
                        if voucher_card == card then
                            card_data.position = i
                            break
                        end
                    end
                end
                
                ActionTracker.log_action("BUY_VOUCHER", card_data)
            end
        end)
    end
    
    -- Buy booster pack
    if G.FUNCS.buy_and_use_consumeable then
        G.FUNCS.buy_and_use_consumeable = Hook.addcallback(G.FUNCS.buy_and_use_consumeable, function(e)
            local card = e.config.ref_table
            
            if card and card:is(Card) then
                local card_data = {
                    key = card.config.center.key,
                    name = card.config and card.config.center and card.config.center.name,
                    cost = card.cost,
                    pack_size = card.ability and card.ability.extra
                }
                
                -- Find position in shop boosters
                if G.shop_booster and G.shop_booster.cards then
                    for i, booster_card in ipairs(G.shop_booster.cards) do
                        if booster_card == card then
                            card_data.position = i
                            break
                        end
                    end
                end
                
                ActionTracker.log_action("BUY_BOOSTER", card_data)
            end
        end)
    end
end

-- Track booster pack decisions
function ActionTracker.hook_booster_actions()
    -- Skip booster pack
    if G.FUNCS.skip_booster then
        G.FUNCS.skip_booster = Hook.addcallback(G.FUNCS.skip_booster, function(e)
            ActionTracker.log_action("SKIP_BOOSTER_PACK", {})
        end)
    end
    
    -- Use card (comprehensive tracking)
    if G.FUNCS.use_card then
        G.FUNCS.use_card = Hook.addcallback(G.FUNCS.use_card, function(e, mute, nosave)
            local card = e.config.ref_table
            
            if card and card:is(Card) then
                local card_data = {
                    key = card.config and card.config.center and card.config.center.key,
                    name = card.config and card.config.center and card.config.center.name,
                    set = card.ability and card.ability.set,
                    cost = card.cost,
                    edition = card.edition,
                    seal = card.seal
                }
                
                -- Determine the action type based on card set and context
                local action_type = nil
                local position = nil
                
                -- Check if it's a booster pack selection
                if G.pack_cards and G.pack_cards.cards then
                    for i, pack_card in ipairs(G.pack_cards.cards) do
                        if pack_card == card then
                            position = i
                            action_type = "SELECT_BOOSTER_CARD"
                            break
                        end
                    end
                end
                
                -- If not from pack, determine by card type and area
                if not action_type then
                    if card.ability.set == 'Booster' then
                        action_type = "USE_BOOSTER_PACK"
                        -- Find position in shop boosters or consumeables
                        if G.shop_booster and G.shop_booster.cards then
                            for i, booster_card in ipairs(G.shop_booster.cards) do
                                if booster_card == card then
                                    position = i
                                    break
                                end
                            end
                        elseif G.consumeables and G.consumeables.cards then
                            for i, consumeable_card in ipairs(G.consumeables.cards) do
                                if consumeable_card == card then
                                    position = i
                                    break
                                end
                            end
                        end
                        
                    elseif card.ability.consumeable then
                        action_type = "USE_CONSUMABLE"
                        -- Find position in consumeables area
                        if G.consumeables and G.consumeables.cards then
                            for i, consumeable_card in ipairs(G.consumeables.cards) do
                                if consumeable_card == card then
                                    position = i
                                    break
                                end
                            end
                        end
                        
                    elseif card.ability.set == 'Voucher' then
                        action_type = "USE_VOUCHER"
                        -- Find position in shop vouchers
                        if G.shop_vouchers and G.shop_vouchers.cards then
                            for i, voucher_card in ipairs(G.shop_vouchers.cards) do
                                if voucher_card == card then
                                    position = i
                                    break
                                end
                            end
                        end
                        
                    elseif card.ability.set == 'Enhanced' or card.ability.set == 'Default' then
                        action_type = "ADD_PLAYING_CARD"
                        -- This is adding a playing card to deck (from booster pack usually)
                        
                    elseif card.ability.set == 'Joker' then
                        action_type = "ADD_JOKER"
                        -- This is adding a joker to collection (from booster pack usually)
                    end
                end
                
                -- Get highlighted hand cards for consumables that target cards
                if card.ability.consumeable or action_type == "USE_CONSUMABLE" then
                    local highlighted_positions = {}
                    if G.hand and G.hand.cards then
                        for j, hand_card in ipairs(G.hand.cards) do
                            if hand_card.highlighted then
                                table.insert(highlighted_positions, j)
                            end
                        end
                    end
                    card_data.highlighted_hand_positions = highlighted_positions
                end
                
                -- Add position if found
                if position then
                    card_data.position = position
                end
                
                -- Log the action if we identified it
                if action_type then
                    ActionTracker.log_action(action_type, card_data)
                else
                    -- Fallback for unknown card usage
                    ActionTracker.log_action("USE_CARD_UNKNOWN", card_data)
                end
            end
        end)
    end
end

-- Track joker selling decisions
-- need to find in blind sell options (action == None)
function ActionTracker.hook_selling_actions()
    if G.FUNCS.sell_card then
        G.FUNCS.sell_card = Hook.addcallback(G.FUNCS.sell_card, function(e)
            sendDebugMessage(tostring(G.FUNCS.sell_card))
            -- Fix: Use e.config.ref_table instead of e.card
            local card = e.config.ref_table
            
            if card and card:is(Card) then
                local card_data = {
                    key = card.config.center.key,
                    name = card.config and card.config.center and card.config.center.name,
                    set = card.ability and card.ability.set,
                    sell_value = card.sell_cost or 0,
                    cost = card.cost or 0
                }
                
                if card.area and card.area.config.type == 'joker' and G.jokers and G.jokers.cards then
                    -- Find position in jokers
                    sendDebugMessage("HERE")
                    for i, joker_card in ipairs(G.jokers.cards) do
                        if joker_card == card then
                            card_data.position = i
                            card_data.area = "jokers"
                            ActionTracker.log_action("SELL_JOKER", card_data)
                            break
                        end
                    end
                elseif card.area and card.area.config.type == 'consumeable' and G.consumeables and G.consumeables.cards then
                    -- Fix: G.consumables -> G.consumeables
                    for i, consumable_card in ipairs(G.consumables.cards) do
                        if consumable_card == card then
                            card_data.position = i
                            card_data.area = "consumeables"
                            ActionTracker.log_action("SELL_CONSUMABLE", card_data)
                            break
                        end
                    end
                end
            end
        end)
    end
end

-- Track consumable usage
-- function ActionTracker.hook_consumable_actions()
--     -- This is tricky because consumables can be used from hand or consumable area
--     -- We'll track when a consumable is activated
--     local original_use_consumable = Card.use_consumeable
--     if original_use_consumable then
--         Card.use_consumeable = function(self, ...)
--             if self.area and self.area.config.type == 'consumeable' and G.consumables and G.consumables.cards then
--                 -- Find position in consumables area
--                 for i, consumable_card in ipairs(G.consumables.cards) do
--                     if consumable_card == self then
--                         ActionTracker.log_action("USE_CONSUMABLE", {i})
--                         break
--                     end
--                 end
--             end
--             return original_use_consumable(self, ...)
--         end
--     end
-- end

-- Track rearrangement actions (these are harder to detect, might need different approach)
function ActionTracker.hook_rearrange_actions()
    local area_states = {
        jokers = {cards = {}, last_update = 0},
        hand = {cards = {}, last_update = 0},
    }

    -- track user input vs game rearrangment
    local user_input_active = false
    local input_timeout = .5
    local last_input_time = 0

    local function capture_area_state(area)
        if not area or not area.cards then return {} end
        local state = {}
        for i, card in ipairs(ara.cards) do
            if card and card.config and card.config.center then
                state[i] = {
                    key = card.config.center.key,
                    sort_id = card.sort_id,
                    unique_id = card.unique_val or tostring(card)
                }
            end
        end
        return state
    end

    -- compare states and detect rearrangments
    local function detect_rearrangement(area_name, area)
        if not area or not area.cards then return end

        local current_state = capture_area_state(area)
        local stored_states = area_states[area_name]
        local prev_state = stored_states.cards

        -- skip if no previous state or empty area
        if #prev_state == 0 or #current_state == 0 then
            area_states[area_name].cards = current_state
            area_states[area_name].last_update = os.time()
            return
        end

        local current_keys = {}
        local prev_keys = {}

        for _, card_info in pairs(current_state) do
            table.insert(current_keys, card_info.key)
        end
        for _, card_info in pairs(prev_state) do
            table.insert(prev_keys, card_info.key)
        end

        table.sort(current_keys)
        table.sort(prev_keys)

        -- only track rearrangments, not additions or removals
        local same_cards = #current_keys == #prev_keys
        if same_cards then
            for i = 1, #current_keys do
                if current_keys[i] ~= prev_keys[i] then
                    same_cards = false
                    break
                end
            end
        end

        if same_cards and #current_state > 1 then
            local changes = {}
            for i =1, #current_state do
                local current_card = current_state[i]
                
                -- find where card was in prev state
                for j = 1, #prev_state do
                    if prev_state[j].unique_id == current_card.unique_id and i ~= j then
                        table.insert(changes, {
                            card_key = current_card.key,
                            from_position = j,
                            to_position = i
                        })
                        break
                    end
                end
            end

            if #changes > 0 then
                -- player or game initiated
                local current_time = os.time()
                local time_since_input = current_time - last_input_time
                local is_player_action = user_input_active or time_since_input < input_timeout

                local action_type = is_player_action and "PLAYER_REARRANGE" or "GAME_REARRANGE"

                for _, change in ipairs(changes) do
                    ActionTracker.log_action(action_type, {
                        area_type = area_name,
                        card_key = change.card_key,
                        from_position = change.from_position,
                        to_position = change.to_position,
                        timestamp = current_time,
                        game_state = G.STATE and ActionTracker.get_state_name(G.STATE) or "UNKNOWN",
                        -- Additional context for ML
                        total_cards_in_area = #current_state,
                        area_capacity = area.config and area.config.card_limit,
                        user_input_detected = is_player_action
                    })
                end
            end
        end

        -- Update stored state
        area_states[area_name].cards = current_state
        area_states[area_name].last_update = os.time()

        if CardArea then
        -- Hook emplace (when cards are placed in areas)
        if CardArea.emplace then
            local original_emplace = CardArea.emplace
            CardArea.emplace = function(self, card, pos, ...)
                local result = original_emplace(self, card, pos, ...)
                
                -- Check for rearrangement after emplace
                if self == G.jokers then
                    detect_rearrangement("jokers", self)
                elseif self == G.hand then
                    detect_rearrangement("hand", self)
                end
                
                return result
            end
        end

        if CardArea.remove_card then
            local original_remove_card = CardArea.remove_card
            CardArea.remove_card = function(self, card, ...)
                local result = original_remove_card(self, card, ...)
                
                -- Don't check immediately on remove, wait for emplace
                -- Just reset input timer
                user_input_active = false
                
                return result
            end
        end

        ActionTracker.check_rearrangements_periodic = function(dt)
        local current_time = love.timer.getTime()
        
        -- Reset user input flag after timeout
        if current_time - last_input_time > input_timeout then
            user_input_active = false
        end
        
        -- Periodic state check
        if current_time - last_periodic_check > PERIODIC_CHECK_INTERVAL then
            last_periodic_check = current_time
            
            -- Only check during states where rearrangement matters
            if G and G.STATE and (
                G.STATE == G.STATES.SELECTING_HAND or
                G.STATE == G.STATES.SHOP or
                G.STATE == G.STATES.BLIND_SELECT or
                G.STATE == G.STATES.DRAW_TO_HAND or
                G.STATE == G.STATES.HAND_PLAYED
            ) then
                detect_rearrangement("jokers", G.jokers)
                detect_rearrangement("hand", G.hand)
            end
        end
    end
    
    -- Hook this into the main update loop
    if BalatrobotAPI and BalatrobotAPI.update then
        local original_update = BalatrobotAPI.update
        BalatrobotAPI.update = function(dt)
            ActionTracker.check_rearrangements_periodic(dt)
            return original_update(dt)
        end
    end
    
    -- Initialize area states
    if G and G.jokers then detect_rearrangement("jokers", G.jokers) end
    if G and G.hand then detect_rearrangement("hand", G.hand) end
    end
end
end

-- Track run start decisions
function ActionTracker.hook_run_start()
    if G.FUNCS.start_run then
        G.FUNCS.start_run = Hook.addcallback(G.FUNCS.start_run, function(e)

            Utils.resetAllIds()
            
            -- The run start data might not be in e directly, need to check G.GAME state
            local stake = G.GAME and G.GAME.stake or 1
            local deck = G.GAME and G.GAME.selected_back and G.GAME.selected_back.name or "Red Deck"
            local seed = G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.seed or nil
            local challenge = G.GAME and G.GAME.challenge and G.GAME.challenge or nil
            
            local run_data = {
                stake = stake,
                deck = deck,
                seed = seed,
                challenge = challenge
            }
            
            sendDebugMessage('Start run with: ' .. tostring(seed))
            ActionTracker.log_action("START_RUN", run_data)
        end)
    end
end

function ActionTracker.get_state_name(state)
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

function ActionTracker.get_all_actions()
    -- maybe update to reset with new blind_id or shop_id
    return ActionTracker.actions
end

function ActionTracker.clear_actions()
    ActionTracker.actions = {}
end

return ActionTracker