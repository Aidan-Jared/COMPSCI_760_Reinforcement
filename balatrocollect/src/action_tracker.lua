ActionTracker = {}
ActionTracker.actions = {}

function ActionTracker.init()
    ActionTracker.actions = {}
    sendDebugMessage("ActionTracker initialized for session: " .. tostring(Utils.current_session_id))
end


function ActionTracker.log_action(action_type, params, card_info)
    
    Utils.ensureSessionId()

    local current_context_ids = Utils.getCurrentContextIds()
    
    local formatted_params = ActionTracker.format_params_for_bot(action_type, params, card_info)

    local action = {
        timestamp = os.time(),
        game_time = BalatrobotAPI.game_start_time and (os.time() - BalatrobotAPI.game_start_time) or 0,
        session_id = current_context_ids.session_id,
        gamestate_id = current_context_ids.gamestate_id,
        action = action_type,
        params = formatted_params,
        card_info = card_info or {},
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

function ActionTracker.format_params_for_bot(action_type, params, card_info)
    local formatted_params = params or {}
    
    -- Map action types to Bot.ACTIONS format
    if action_type == "PLAY_HAND" or action_type == "DISCARD_HAND" then
        -- Bot expects: action_type, [card_positions]
        if type(params) == "table" and #params > 0 then
            formatted_params = params -- params should already be card positions
        end
        
    elseif action_type == "SELECT_BLIND" or action_type == "SKIP_BLIND" then
        -- Bot expects: action_type (no additional params needed)
        formatted_params = {}
        
    elseif action_type == "BUY_CARD" or action_type == "BUY_VOUCHER" or action_type == "BUY_BOOSTER" then
        -- Bot expects: action_type, [position]
        if type(params) == "number" then
            formatted_params = {params}
        elseif type(params) == "table" and params[1] then
            formatted_params = {params[1]}
        else
            formatted_params = {1} -- default to position 1
        end
        
    elseif action_type == "SELECT_BOOSTER_CARD" then
        -- Bot expects: action_type, [hand_positions], [booster_position]
        local hand_positions = {}
        local booster_position = 1
        
        if card_info and card_info.highlighted_hand_positions then
            hand_positions = card_info.highlighted_hand_positions
        end
        
        if type(params) == "number" then
            booster_position = params
        elseif type(params) == "table" and params[1] then
            booster_position = params[1]
        end
        
        formatted_params = {hand_positions, {booster_position}}
        
    elseif action_type == "SELL_JOKER" then
        -- Bot expects: action_type, [position]
        if type(params) == "number" then
            formatted_params = {params}
        elseif type(params) == "table" and params[1] then
            formatted_params = {params[1]}
        else
            formatted_params = {}
        end
        
    elseif action_type == "REARRANGE_JOKERS" or action_type == "REARRANGE_CONSUMABLES" or action_type == "REARRANGE_HAND" then
        -- Bot expects: action_type, [new_order_array]
        -- params should be the new order array from hook_rearrange_actions
        if type(params) == "table" and #params > 0 then
            -- params is already the new_order array like {2, 1, 4, 3}
            formatted_params = {params}
        else
            -- Fallback: create identity order if no valid params
            formatted_params = {{}}
        end
        
    elseif action_type == "REARRANGE_CARD" then
        -- Legacy individual rearrange action (if still used elsewhere)
        -- Bot expects: action_type, [from_position], [to_position], [area_type]
        if type(params) == "table" then
            local from_pos = params.from_position or 1
            local to_pos = params.to_position or 1
            local area_type = params.area_type or "hand"
            formatted_params = {from_pos, to_pos, area_type}
        else
            formatted_params = {1, 1, "hand"}
        end
        
    elseif action_type == "START_RUN" then
        -- Bot expects: action_type, [stake], [deck], [seed], [challenge]
        local stake = {1}
        local deck = {"Red Deck"}
        local seed = {G.GAME.pseudorandom.seed}
        local challenge = {nil}
        
        if type(params) == "table" then
            if params.stake then stake = {params.stake} end
            if params.deck then deck = {params.deck} end
            if params.seed then seed = {params.seed} end
            if params.challenge then challenge = {params.challenge} end
        end
        
        formatted_params = {stake, deck, seed, challenge}
        
    else
        -- Default case - keep params as is
        if type(params) ~= "table" then
            formatted_params = params and {params} or {}
        end
    end
    
    return formatted_params
end

-- Convert ActionTracker action types to Bot.ACTIONS constants
function ActionTracker.get_bot_action_constant(action_type)
    local action_map = {
        ["SELECT_BLIND"] = 1,     -- Bot.ACTIONS.SELECT_BLIND
        ["SKIP_BLIND"] = 2,       -- Bot.ACTIONS.SKIP_BLIND  
        ["PLAY_HAND"] = 3,        -- Bot.ACTIONS.PLAY_HAND
        ["DISCARD_HAND"] = 4,     -- Bot.ACTIONS.DISCARD_HAND
        ["END_SHOP"] = 5,         -- Bot.ACTIONS.END_SHOP
        ["REROLL_SHOP"] = 6,      -- Bot.ACTIONS.REROLL_SHOP
        ["BUY_CARD"] = 7,         -- Bot.ACTIONS.BUY_CARD
        ["BUY_VOUCHER"] = 8,      -- Bot.ACTIONS.BUY_VOUCHER
        ["BUY_BOOSTER"] = 9,      -- Bot.ACTIONS.BUY_BOOSTER
        ["SELECT_BOOSTER_CARD"] = 10, -- Bot.ACTIONS.SELECT_BOOSTER_CARD
        ["SKIP_BOOSTER_PACK"] = 11,   -- Bot.ACTIONS.SKIP_BOOSTER_PACK
        ["SELL_JOKER"] = 12,      -- Bot.ACTIONS.SELL_JOKER
        ["USE_CONSUMABLE"] = 13,  -- Bot.ACTIONS.USE_CONSUMABLE
        ["SELL_CONSUMABLE"] = 14, -- Bot.ACTIONS.SELL_CONSUMABLE
        ["REARRANGE_JOKERS"] = 15,    -- Bot.ACTIONS.REARRANGE_JOKERS
        ["REARRANGE_CONSUMABLES"] = 16, -- Bot.ACTIONS.REARRANGE_CONSUMABLES
        ["REARRANGE_HAND"] = 17,  -- Bot.ACTIONS.REARRANGE_HAND
        ["START_RUN"] = 19,       -- Bot.ACTIONS.START_RUN
    }
    
    return action_map[action_type] or nil
end

-- Create bot-compatible action format
function ActionTracker.create_bot_action(action_type, params, card_info)
    local bot_action_constant = ActionTracker.get_bot_action_constant(action_type)
    if not bot_action_constant then
        return nil
    end
    
    local formatted_params = ActionTracker.format_params_for_bot(action_type, params, card_info)
    local bot_action = {bot_action_constant}
    
    -- Add formatted parameters
    if type(formatted_params) == "table" and #formatted_params > 0 then
        for i, param in ipairs(formatted_params) do
            table.insert(bot_action, param)
        end
    elseif formatted_params then
        table.insert(bot_action, formatted_params)
    end
    
    return bot_action
end


-- Clear actions when starting new context sessions
function ActionTracker.clear_actions_for_new_context(context_type)
    sendDebugMessage("Clearing actions for new " .. context_type .. " context")
    ActionTracker.actions = {}
end


function ActionTracker.get_hand_score_info()
    local score_info = {
        hand_name = "Unknown",
        chips = 0,
        mult = 0,
        total_score = 0,
        hand_level = 0
    }
    
    -- Try to get the last evaluated hand from game state
    if G and G.GAME then
        -- Get current round scoring
        if G.GAME.current_round then
            score_info.chips = G.GAME.current_round.chips_total or 0
            score_info.mult = G.GAME.current_round.mult_total or 0
            score_info.total_score = score_info.chips * score_info.mult
        end
        
        -- Get the last played hand name and level
        if G.GAME.last_hand_played then
            score_info.hand_name = G.GAME.last_hand_played
            if G.GAME.hands and G.GAME.hands[score_info.hand_name] then
                score_info.hand_level = G.GAME.hands[score_info.hand_name].level or 0
            end
        end
    end
    
    return score_info
end

-- Helper to detect hand type from highlighted cards (fallback method)
function ActionTracker.detect_hand_type(highlighted_positions)
    local hand_info = {
        hand_name = "High Card",
        cards_used = {}
    }
    
    if not G or not G.hand or not G.hand.cards or not highlighted_positions then
        return hand_info
    end
    
    -- Get the highlighted cards
    local played_cards = {}
    for _, pos in ipairs(highlighted_positions) do
        if G.hand.cards[pos] then
            local card = G.hand.cards[pos]
            table.insert(played_cards, card)
            table.insert(hand_info.cards_used, {
                position = pos,
                rank = card.base and card.base.value or "unknown",
                suit = card.base and card.base.suit or "unknown",
                enhancement = card.config and card.config.center and card.config.center.key or nil,
                edition = card.edition and card.edition.type or nil,
                seal = card.seal and card.seal.key or nil
            })
        end
    end
    
    if #played_cards == 0 then
        return hand_info
    end
    
    -- Basic hand type detection
    local rank_counts = {}
    local suits = {}
    
    for _, card in ipairs(played_cards) do
        if card.base then
            local rank = card.base.value
            rank_counts[rank] = (rank_counts[rank] or 0) + 1
            
            local suit = card.base.suit
            suits[suit] = (suits[suit] or 0) + 1
        end
    end
    
    -- Count pairs, trips, etc.
    local pairs_c = 0 -- different name to not override functions
    local trips = false
    local quads = false
    local fives = false
    
    for rank, count in pairs(rank_counts) do
        if count == 2 then pairs_c = pairs_c + 1 end
        if count == 3 then trips = true end
        if count == 4 then quads = true end
        if count == 5 then fives = true end
    end
    
    -- Check for flush
    local is_flush = false
    for suit, count in pairs(suits) do
        if count >= 5 then is_flush = true break end
    end
    
    -- Determine hand type (simplified logic)
    if fives then
        hand_info.hand_name = "Five of a Kind"
    elseif is_flush and #played_cards >= 5 then
        -- Could be straight flush, but we'll simplify
        hand_info.hand_name = "Flush"
    elseif quads then
        hand_info.hand_name = "Four of a Kind"
    elseif trips and pairs_c >= 1 then
        hand_info.hand_name = "Full House"
    elseif is_flush then
        hand_info.hand_name = "Flush"
    elseif trips then
        hand_info.hand_name = "Three of a Kind"
    elseif pairs_c >= 2 then
        hand_info.hand_name = "Two Pair"
    elseif pairs_c >= 1 then
        hand_info.hand_name = "Pair"
    else
        hand_info.hand_name = "High Card"
    end
    
    return hand_info
end

-- track finished hand decisioins
function ActionTracker.hook_hand_actions()
    -- Play button - capture action immediately, then wait for scoring
    if G.FUNCS.play_cards_from_highlighted then
        G.FUNCS.play_cards_from_highlighted = Hook.addcallback(G.FUNCS.play_cards_from_highlighted, function (e)
            local card_positions = {}
            if G.hand then
                for i, card in ipairs(G.hand.cards) do
                    if card.highlighted then
                        table.insert(card_positions, i)
                    end
                end
            end

            if #card_positions > 0 then
                -- Get hand type immediately
                local hand_detection = ActionTracker.detect_hand_type(card_positions)
                
                -- Store initial action data
                local initial_action_data = {
                    card_positions = card_positions,
                    predicted_hand = hand_detection.hand_name,
                    cards_used = hand_detection.cards_used,
                    pre_play_state = {
                        chips = G.GAME.chips or 0,
                        dollars = G.GAME.dollars or 0,
                        discards_left = G.GAME.current_round and G.GAME.current_round.discards_left or 0,
                        hands_left = G.GAME.current_round and G.GAME.current_round.hands_left or 0
                    }
                }
                
                -- Queue a delayed action to capture the final scoring
                G.E_MANAGER:add_event(Event({
                    trigger = 'after',
                    delay = 0.5,  -- Wait for hand evaluation to complete
                    blocking = false,
                    func = function()
                        -- Now get the actual scoring results
                        local score_info = ActionTracker.get_hand_score_info()
                        
                        local final_action_data = {
                            cards_used = hand_detection.cards_used,
                            hand_played = {
                                name = score_info.hand_name,
                                level = score_info.hand_level,
                                chips = score_info.chips,
                                mult = score_info.mult,
                                total_score = score_info.total_score
                            },
                            game_state_after = {
                                chips = G.GAME.chips or 0,
                                dollars = G.GAME.dollars or 0,
                                discards_left = G.GAME.current_round and G.GAME.current_round.discards_left or 0,
                                hands_left = G.GAME.current_round and G.GAME.current_round.hands_left or 0
                            }
                        }
                        
                        ActionTracker.log_action("PLAY_HAND", card_positions, final_action_data)
                        sendDebugMessage("Played " .. score_info.hand_name .. 
                                       " for " .. score_info.total_score .. " points")
                        return true
                    end
                }))
            end
        end)
    end

    -- Discard button - simpler since no scoring involved
    if G.FUNCS.discard_cards_from_highlighted then
        G.FUNCS.discard_cards_from_highlighted = Hook.addcallback(G.FUNCS.discard_cards_from_highlighted, function (e)
            local card_positions = {}
            local discarded_cards = {}
            
            if G.hand then
                for i, card in ipairs(G.hand.cards) do
                    if card.highlighted then
                        table.insert(card_positions, i)
                        table.insert(discarded_cards, {
                            position = i,
                            rank = card.base and card.base.value or "unknown",
                            suit = card.base and card.base.suit or "unknown",
                            enhancement = card.config and card.config.center and card.config.center.key or nil,
                            edition = card.edition and card.edition.type or nil,
                            seal = card.seal and card.seal.key or nil
                        })
                    end
                end
            end
            
            if #card_positions > 0 then
                local action_data = {
                    discarded_cards = discarded_cards,
                    game_state = {
                        discards_left = G.GAME.current_round and G.GAME.current_round.discards_left or 0,
                        hands_left = G.GAME.current_round and G.GAME.current_round.hands_left or 0,
                        dollars = G.GAME.dollars or 0
                    }
                }
                
                ActionTracker.log_action("DISCARD_HAND", card_positions, action_data)
                sendDebugMessage("Discarded " .. #card_positions .. " cards")
            end
        end)
    end
end

function ActionTracker.hook_hand_evaluation()
    -- Hook the actual hand evaluation function if it exists
    if G and G.FUNCS and G.FUNCS.evaluate_play then
        G.FUNCS.evaluate_play = Hook.addcallback(G.FUNCS.evaluate_play, function(...)
            -- This will be called after hand evaluation
            local score_info = ActionTracker.get_hand_score_info()
            sendDebugMessage("Hand evaluated: " .. score_info.hand_name .. 
                           " scored " .. score_info.total_score)
        end)
    end
    
    -- Also try hooking the scoring calculation
    if G and G.GAME and G.GAME.blind and G.GAME.blind.chips then
        -- This is more complex and might need adjustment based on game internals
    end
end

-- Track blind selection decisioins
function ActionTracker.hook_blind_actions()
    -- Select blind
    if G.FUNCS.select_blind then
        G.FUNCS.select_blind = Hook.addcallback(G.FUNCS.select_blind, function(e)
            ActionTracker.log_action("SELECT_BLIND", {}, {})
        end)
    end
    
    -- Skip blind
    if G.FUNCS.skip_blind then
        G.FUNCS.skip_blind = Hook.addcallback(G.FUNCS.skip_blind, function(e)
            ActionTracker.log_action("SKIP_BLIND", {}, {})
        end)
    end
end

function ActionTracker.hook_shop_actions()
    -- Shop reroll
    if G.FUNCS.reroll_shop then
        G.FUNCS.reroll_shop = Hook.addcallback(G.FUNCS.reroll_shop, function(e)
            ActionTracker.log_action("REROLL_SHOP", {}, {})
        end)
    end
    
    -- End shop
    if G.FUNCS.toggle_shop then
        G.FUNCS.toggle_shop = Hook.addcallback(G.FUNCS.toggle_shop, function(e)
            ActionTracker.log_action("END_SHOP", {}, {})
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
                    ActionTracker.log_action("BUY_CARD", position, card_data)
                else
                    -- Log anyway with available data
                    ActionTracker.log_action("BUY_CARD", position, card_data)
                end
                
                sendDebugMessage('Bought card: ' .. tostring(card_data.name or card_data.key or "unknown"))
            else
                sendDebugMessage('Could not get card data from buy_from_shop event')
            end
        end)
    end
    
    -- Buy voucher
    if G.FUNCS.buy_voucher then
        local position = nil
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
                            position = i
                            break
                        end
                    end
                end
                
                ActionTracker.log_action("BUY_VOUCHER", position, card_data)
            end
        end)
    end
    
    -- Buy booster pack
    if G.FUNCS.buy_and_use_consumeable then
        local position = nil
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
                            position = i
                            break
                        end
                    end
                end
                
                ActionTracker.log_action("BUY_BOOSTER", position, card_data)
            end
        end)
    end
end

-- Track booster pack decisions
function ActionTracker.hook_booster_actions()
    -- Skip booster pack
    if G.FUNCS.skip_booster then
        G.FUNCS.skip_booster = Hook.addcallback(G.FUNCS.skip_booster, function(e)
            ActionTracker.log_action("SKIP_BOOSTER_PACK", {}, {})
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
                
                -- Log the action if we identified it
                if action_type then
                    ActionTracker.log_action(action_type, position, card_data)
                else
                    -- Fallback for unknown card usage
                    ActionTracker.log_action("USE_CARD_UNKNOWN", position, card_data)
                end
            end
        end)
    end
end

-- Track joker selling decisions
-- need to find in blind sell options (action == None)
function ActionTracker.hook_selling_actions()
    if G.FUNCS.sell_card then
        local position = nil
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
                    for i, joker_card in ipairs(G.jokers.cards) do
                        if joker_card == card then
                            position = i
                            card_data.area = "jokers"
                            ActionTracker.log_action("SELL_JOKER", position, card_data)
                            break
                        end
                    end
                elseif card.area and card.area.config.type == 'consumeable' and G.consumeables and G.consumeables.cards then
                    -- Fix: G.consumables -> G.consumeables
                    for i, consumable_card in ipairs(G.consumables.cards) do
                        if consumable_card == card then
                            position = i
                            card_data.area = "consumeables"
                            ActionTracker.log_action("SELL_CONSUMABLE", position, card_data)
                            break
                        end
                    end
                end
            end
        end)
    end
end


-- Track rearrangement actions (these are harder to detect, might need different approach)

-- Enhanced card rearrangement tracking for ActionTracker
-- This replaces the existing hook_rearrange_actions function

function ActionTracker.hook_rearrange_actions()
    -- Track area states for comparison
    local area_states = {
        hand = {cards = {}, last_positions = {}},
        jokers = {cards = {}, last_positions = {}},
    }
    
    -- Capture current card positions and identities
    local function capture_area_state(area, area_name)
        if not area or not area.cards then return {} end
        
        local state = {}
        local positions = {}
        
        for i, card in ipairs(area.cards) do
            if card and card.config and card.config.center then
                -- Create unique identifier for card
                local card_id = card.config.center.key .. "_" .. (card.unique_val or tostring(card))
                
                state[i] = {
                    card_id = card_id,
                    key = card.config.center.key,
                    name = card.config.center.name or card.config.center.key,
                    position = i,
                    -- Additional card info for context
                    suit = card.config and card.config.card and card.config.card.suit,
                    value = card.config and card.config.card and card.config.card.value,
                    set = card.ability and card.ability.set
                }
                positions[card_id] = i
            end
        end
        
        return {cards = state, positions = positions}
    end
    
    -- Detect rearrangements by comparing positions
    local function detect_rearrangement(area, area_name)
        if not area or not area.cards then return end
        
        local current_state = capture_area_state(area, area_name)
        local previous_state = area_states[area_name]
        
        -- Skip if no previous state or different card count
        if not previous_state.cards or #current_state.cards == 0 or 
           #current_state.cards ~= #previous_state.cards then
            area_states[area_name] = current_state
            return
        end
        
        -- Check if same cards exist (no additions/removals)
        local current_card_ids = {}
        local previous_card_ids = {}
        
        for _, card_info in pairs(current_state.cards) do
            table.insert(current_card_ids, card_info.card_id)
        end
        for _, card_info in pairs(previous_state.cards) do
            table.insert(previous_card_ids, card_info.card_id)
        end
        
        table.sort(current_card_ids)
        table.sort(previous_card_ids)
        
        -- Only proceed if same cards (just potentially reordered)
        local same_cards = #current_card_ids == #previous_card_ids
        if same_cards then
            for i = 1, #current_card_ids do
                if current_card_ids[i] ~= previous_card_ids[i] then
                    same_cards = false
                    break
                end
            end
        end
        
        if same_cards and #current_state.cards > 1 then
            -- Build complete rearrangement order for bot compatibility
            local new_order = {}
            for i = 1, #current_state.cards do
                local current_card = current_state.cards[i]
                if current_card then
                    local original_position = previous_state.positions[current_card.card_id]
                    if original_position then
                        new_order[original_position] = i
                    end
                end
            end
            
            -- Check if any actual reordering occurred
            local has_changes = false
            for i = 1, #new_order do
                if new_order[i] ~= i then
                    has_changes = true
                    break
                end
            end
            
            if has_changes then
                local action_type = area_name == "hand" and "REARRANGE_HAND" or 
                                 area_name == "jokers" and "REARRANGE_JOKERS" or 
                                 "REARRANGE_CONSUMABLES"
                
                ActionTracker.log_action(action_type, new_order, {
                    area_type = area_name,
                    total_cards = #current_state.cards,
                    timestamp = os.time()
                })
                
                sendDebugMessage("Detected " .. area_name .. " rearrangement: " .. table.concat(new_order, ","))
            end
        end
        
        -- Update stored state
        area_states[area_name] = current_state
    end
    
    -- Hook into CardArea:emplace to detect when cards are placed
    local original_emplace = CardArea.emplace
    CardArea.emplace = function(self, card, location, stay_flipped)
        local result = original_emplace(self, card, location, stay_flipped)
        
        -- Only track hand and jokers areas during relevant states
        if (self == G.hand or self == G.jokers) and G.STATE then
            -- Avoid tracking during automatic game operations
            if G.STATE == G.STATES.SELECTING_HAND or 
               G.STATE == G.STATES.SHOP or 
               G.STATE == G.STATES.BLIND_SELECT then
                
                -- Use event system to check after position is finalized
                G.E_MANAGER:add_event(Event({
                    trigger = 'after',
                    delay = 0.05, -- Small delay to ensure position is set
                    blocking = false,
                    func = function()
                        local area_name = (self == G.hand) and "hand" or "jokers"
                        detect_rearrangement(self, area_name)
                        return true
                    end
                }))
            end
        end
        
        return result
    end
    
    -- Hook into CardArea:align_cards which is called after positioning
    local original_align_cards = CardArea.align_cards
    CardArea.align_cards = function(self)
        local result = original_align_cards(self)
        
        -- Track position changes after alignment
        if (self == G.hand or self == G.jokers) and G.STATE then
            if G.STATE == G.STATES.SELECTING_HAND or 
               G.STATE == G.STATES.SHOP or 
               G.STATE == G.STATES.BLIND_SELECT then
                
                local area_name = (self == G.hand) and "hand" or "jokers"
                
                -- Use timer to avoid too frequent checks
                G.E_MANAGER:add_event(Event({
                    trigger = 'after',
                    delay = 0.1,
                    blocking = false,
                    func = function()
                        detect_rearrangement(self, area_name)
                        return true
                    end
                }))
            end
        end
        
        return result
    end
    
    -- Hook into CardArea:set_ranks which is called when card order changes
    local original_set_ranks = CardArea.set_ranks
    CardArea.set_ranks = function(self)
        local result = original_set_ranks(self)
        
        -- Check for rearrangements after ranks are set
        if (self == G.hand or self == G.jokers) and G.STATE then
            if G.STATE == G.STATES.SELECTING_HAND or 
               G.STATE == G.STATES.SHOP or 
               G.STATE == G.STATES.BLIND_SELECT then
                
                local area_name = (self == G.hand) and "hand" or "jokers"
                
                -- Immediate check since set_ranks indicates a definitive change
                G.E_MANAGER:add_event(Event({
                    trigger = 'after',
                    delay = 0.02,
                    blocking = false,
                    func = function()
                        detect_rearrangement(self, area_name)
                        return true
                    end
                }))
            end
        end
        
        return result
    end
    
    -- Initialize states for existing areas
    if G and G.hand then
        area_states.hand = capture_area_state(G.hand, "hand")
        sendDebugMessage("Initialized hand rearrangement tracking")
    end
    
    if G and G.jokers then
        area_states.jokers = capture_area_state(G.jokers, "jokers")
        sendDebugMessage("Initialized jokers rearrangement tracking")
    end
    
    -- Periodic state check as backup
    local last_periodic_check = 0
    local PERIODIC_CHECK_INTERVAL = 2.0 -- Check every 2 seconds
    
    local function periodic_check()
        local current_time = love.timer.getTime()
        
        if current_time - last_periodic_check > PERIODIC_CHECK_INTERVAL then
            last_periodic_check = current_time
            
            -- Only check during relevant states
            if G and G.STATE and (
                G.STATE == G.STATES.SELECTING_HAND or
                G.STATE == G.STATES.SHOP or
                G.STATE == G.STATES.BLIND_SELECT
            ) then
                if G.hand then
                    detect_rearrangement(G.hand, "hand")
                end
                if G.jokers then
                    detect_rearrangement(G.jokers, "jokers")
                end
            end
        end
    end
    
    -- Hook periodic check into update loop
    if BalatrobotAPI and BalatrobotAPI.update then
        local original_update = BalatrobotAPI.update
        BalatrobotAPI.update = function(dt)
            periodic_check()
            return original_update(dt)
        end
        sendDebugMessage("Hooked periodic rearrangement check into API update")
    end
    
    sendDebugMessage("Card rearrangement tracking initialized successfully")
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
            ActionTracker.log_action("START_RUN", run_data, {})
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