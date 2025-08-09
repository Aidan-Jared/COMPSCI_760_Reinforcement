ActionTracker = {}
ActionTracker.actions = {}
ActionTracker.session_id = nil

function ActionTracker.init()
    ActionTracker.actions = {}
    ActionTracker.session_id = BalatrobotAPI.game_session_id
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

    table.insert(ActionTracker.actions, action)
    sendDebugMessage("Action logged: " .. action_type)

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
    -- select blind
    if G.FUNCS.select_blind then
        G.FUNCS.select_blind = Hook.addcallback(G.FUNCS.skips_blinds, function (e)
            ActionTracker.log_action("REROLL_SHOP", {})
        end)
    end

    --End Shop
    if G.FUNCS.toggle_shop then
        G.FUNCS.toggle_shop = Hook.addcallback(G.FUNCS.toggle_shop, function (e)
            local card =  e and e.card
            if card and G.shop_jokers and G.shop_jokers.cards then
                for i, shop_card in ipairs(G.shop_jokers.cards) do
                    if shop_card == card then
                        ActionTracker.log_action("BUY_CARD", {i})
                        break
                    end
                end
            end
        end)
    end

    -- Buy Voucher
    if G.FUNCS.buy_voucher then
        G.FUNCS.buy_voucher = Hook.addcallback(G.FUNCS.buy_voucher, function (e)
            local card = e and e.card
            if card and G.shop_vouchers and G.shop_vouchers.cards then
                for i, voucher_card in ipairs(G.shop_vouchers.cards) do
                    if voucher_card == card then
                        ActionTracker.log_action("BUY_VOUCHER", {i})
                        break
                    end
                end
            end
        end)
    end

    -- buy booster pack
    if G.FUNCS.buy_and_use_consumeable then
        G.FUNCS.buy_and_use_consumeable = Hook.addcallback(G.FUNCS.buy_and_use_consumeable, function(e)
            local card = e and e.card
            if card and G.shop_booster and G.shop_booster.cards then
                -- Find position in shop boosters
                for i, booster_card in ipairs(G.shop_booster.cards) do
                    if booster_card == card then
                        ActionTracker.log_action("BUY_BOOSTER", {i})
                        break
                    end
                end
            end
        end)
    end
end

function ActionTracker.hook_booster_actions()
    -- skip booster pack
    if G.FUNCS.skip_booster then
        G.FUNCS.skip_booster = Hook.addcallback(G.FUNCS.skip_booster, function(e)
            ActionTracker.log_action("SKIP_BOOSTER_PACK", {})
        end)
    end

    -- select booster card
    if G.FUNCS.use_card then
        G.FUNCS.use_card = Hook.addcallback(G.FUNCS.use_card, function(e)
            local card = e and e.card
            if card and G.pack_cards and G.pack_cards.cards then
                -- Check if this is a pack selection
                for i, pack_card in ipairs(G.pack_cards.cards) do
                    if pack_card == card then
                        -- Get highlighted hand cards for consumable cards
                        local highlighted_positions = {}
                        if G.hand then
                            for j, hand_card in ipairs(G.hand.cards) do
                                if hand_card.highlighted then
                                    table.insert(highlighted_positions, j)
                                end
                            end
                        end
                        
                        ActionTracker.log_action("SELECT_BOOSTER_CARD", {i, highlighted_positions})
                        break
                    end
                end
            end
        end)
    end
end

-- Track joker selling decisions
function ActionTracker.hook_selling_actions()
    if G.FUNCS.sell_card then
        G.FUNCS.sell_card = Hook.addcallback(G.FUNCS.sell_card, function(e)
            local card = e and e.card
            if card and card.area and card.area.config.type == 'joker' and G.jokers and G.jokers.cards then
                -- Find position in jokers
                for i, joker_card in ipairs(G.jokers.cards) do
                    if joker_card == card then
                        ActionTracker.log_action("SELL_JOKER", {i})
                        break
                    end
                end
            elseif card and card.area and card.area.config.type == 'consumeable' and G.consumables and G.consumables.cards then
                -- Find position in consumables
                for i, consumable_card in ipairs(G.consumables.cards) do
                    if consumable_card == card then
                        ActionTracker.log_action("SELL_CONSUMABLE", {i})
                        break
                    end
                end
            end
        end)
    end
end


-- Track consumable usage
function ActionTracker.hook_consumable_actions()
    -- This is tricky because consumables can be used from hand or consumable area
    -- We'll track when a consumable is activated
    local original_use_consumable = Card.use_consumeable
    if original_use_consumable then
        Card.use_consumeable = function(self, ...)
            if self.area and self.area.config.type == 'consumeable' and G.consumables and G.consumables.cards then
                -- Find position in consumables area
                for i, consumable_card in ipairs(G.consumables.cards) do
                    if consumable_card == self then
                        ActionTracker.log_action("USE_CONSUMABLE", {i})
                        break
                    end
                end
            end
            return original_use_consumable(self, ...)
        end
    end
end

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
            local stake = e and e.stake or 1
            local deck = G.GAME and G.GAME.selected_back and G.GAME.selected_back.name or "Red Deck"
            local seed = e and e.seed or nil
            local challenge = e and e.challenge and e.challenge.name or nil
            
            ActionTracker.log_action("START_RUN", {stake, deck, seed, challenge})
        end)
    end
end

function ActionTracker.get_all_actions()
    return ActionTracker.actions
end

function ActionTracker.clear_actions()
    ActionTracker.actions = {}
end

return ActionTracker