ActionTracker = {}
ActionTracker.actions = {}
ActionTracker.session_id = nil

ActionTracker.last_blind_id = nil
ActionTracker.last_shop_id = nil

function ActionTracker.init()
    ActionTracker.actions = {}
    ActionTracker.session_id = BalatrobotAPI.game_session_id
    ActionTracker.last_blind_id = nil
    ActionTracker.last_shop_id = nil
end

-- Store the last generated IDs from gamestate broadcasts
function ActionTracker.set_last_context_ids(blind_id, shop_id)
   if blind_id then
        ActionTracker.last_blind_id = blind_id
    end
    if shop_id then
        ActionTracker.last_shop_id = shop_id
    end
end

-- Clear context IDs when moving to unrelated states
function ActionTracker.clear_context_ids()
    ActionTracker.last_blind_id = nil
    ActionTracker.last_shop_id = nil
end

-- Clear specific context ID when it's no longer relevant
function ActionTracker.clear_blind_context()
    ActionTracker.last_blind_id = nil
end

function ActionTracker.clear_shop_context()
    ActionTracker.last_shop_id = nil
end

function ActionTracker.log_action(action_type, params)
    local action = {
        timestamp = os.time(),
        game_time = BalatrobotAPI.game_start_time and (os.time() - BalatrobotAPI.game_start_time) or 0,
        session_id = ActionTracker.session_id,
        action = action_type,
        params = params or {},
        game_state = G.STATE and BalatrobotAPI.get_state_name(G.STATE) or "UNKNOWN",
    }

    -- Use the most relevant context ID based on current state
    -- Blind ID takes precedence in blind-related states
    if G and G.STATE and (G.STATE == G.STATES.BLIND_SELECT or G.STATE == G.STATES.SELECTING_HAND or G.STATE == G.STATES.HAND_PLAYED) and ActionTracker.last_blind_id then
        action.context_id = ActionTracker.last_blind_id
        action.context_type = "new_blind"
    -- Shop ID for shop-related states  
    elseif G and G.STATE and G.STATE == G.STATES.SHOP and ActionTracker.last_shop_id then
        action.context_id = ActionTracker.last_shop_id
        action.context_type = "shop_visit"
    -- Fallback to any available context ID
    elseif ActionTracker.last_shop_id then
        action.context_id = ActionTracker.last_shop_id
        action.context_type = "shop_visit"
    elseif ActionTracker.last_blind_id then
        action.context_id = ActionTracker.last_blind_id
        action.context_type = "blind_selection"
    end

    table.insert(ActionTracker.actions, action)
    sendDebugMessage("Action logged: " .. action_type .. " with context: " .. tostring(action.context_id or "none"))

    BalatrobotAPI.broadcast_action(action)
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
            local card = e and e.config and e.config.ref_table
            
            if card and card:is(card) then
                local card_data = {
                    key = card.config and card.config.center and card.config.center.key,
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
            local card = e and e.config and e.config.ref_table
            
            if card and card:is(Card) then
                local card_data = {
                    key = card.config and card.config.center and card.config.center.key,
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
            local card = e and e.config and e.config.ref_table
            
            if card and card:is(Card) then
                local card_data = {
                    key = card.config and card.config.center and card.config.center.key,
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
    
    -- Select booster card (FIXED)
    if G.FUNCS.use_card then
        G.FUNCS.use_card = Hook.addcallback(G.FUNCS.use_card, function(e, mute, nosave)
            -- Fix: Use e.config.ref_table instead of e.card
            local card = e and e.config and e.config.ref_table
            
            if card and card:is(Card) and G.pack_cards and G.pack_cards.cards then
                -- Check if this is a pack selection (card is in pack_cards area)
                for i, pack_card in ipairs(G.pack_cards.cards) do
                    if pack_card == card then
                        local card_data = {
                            key = card.config and card.config.center and card.config.center.key,
                            name = card.config and card.config.center and card.config.center.name,
                            set = card.ability and card.ability.set,
                            position = i
                        }
                        
                        -- Get highlighted hand cards for consumable cards
                        local highlighted_positions = {}
                        if G.hand and G.hand.cards then
                            for j, hand_card in ipairs(G.hand.cards) do
                                if hand_card.highlighted then
                                    table.insert(highlighted_positions, j)
                                end
                            end
                        end
                        
                        card_data.highlighted_hand_positions = highlighted_positions
                        ActionTracker.log_action("SELECT_BOOSTER_CARD", card_data)
                        break
                    end
                end
            elseif card and card:is(Card) and card.area and card.area.config.type == 'consumeable' then
                -- This is a consumable being used from the consumeables area
                if G.consumeables and G.consumeables.cards then
                    for i, consumable_card in ipairs(G.consumeables.cards) do
                        if consumable_card == card then
                            local card_data = {
                                key = card.config and card.config.center and card.config.center.key,
                                name = card.config and card.config.center and card.config.center.name,
                                set = card.ability and card.ability.set,
                                position = i
                            }
                            
                            -- Get highlighted hand cards
                            local highlighted_positions = {}
                            if G.hand and G.hand.cards then
                                for j, hand_card in ipairs(G.hand.cards) do
                                    if hand_card.highlighted then
                                        table.insert(highlighted_positions, j)
                                    end
                                end
                            end
                            
                            card_data.highlighted_hand_positions = highlighted_positions
                            ActionTracker.log_action("USE_CONSUMABLE", card_data)
                            break
                        end
                    end
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
            -- Fix: Use e.config.ref_table instead of e.card
            local card = e and e.config and e.config.ref_table
            
            if card and card:is(Card) then
                local card_data = {
                    key = card.config and card.config.center and card.config.center.key,
                    name = card.config and card.config.center and card.config.center.name,
                    set = card.ability and card.ability.set,
                    sell_value = card.sell_cost or 0,
                    cost = card.cost or 0
                }
                
                if card.area and card.area.config.type == 'joker' and G.jokers and G.jokers.cards then
                    -- Find position in jokers
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
    -- Rearrangement is complex because it involves drag/drop
    -- For now, we'll just log when areas are reordered
    -- This might need to be enhanced based on how the game handles rearrangement
    
    -- Hook joker area reordering
    if G.jokers then
        local original_set_ranks = G.jokers.set_ranks
        if original_set_ranks then
            G.jokers.set_ranks = function(self, ...)
                -- Get current order
                local current_order = {}
                for i, card in ipairs(self.cards) do
                    current_order[i] = i  -- This is simplified
                end
                
                -- Only log if this was a player-initiated reorder
                -- (This is hard to detect, might need additional context)
                
                return original_set_ranks(self, ...)
            end
        end
    end
end

-- Track run start decisions
function ActionTracker.hook_run_start()
    if G.FUNCS.start_run then
        G.FUNCS.start_run = Hook.addcallback(G.FUNCS.start_run, function(e)
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

function ActionTracker.get_all_actions()
    -- maybe update to reset with new blind_id or shop_id
    return ActionTracker.actions
end

function ActionTracker.clear_actions()
    ActionTracker.actions = {}
end

return ActionTracker