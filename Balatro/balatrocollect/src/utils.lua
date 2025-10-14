
Utils = { }

Utils.current_session_id = nil
Utils.current_gamestate_id = nil
Utils.previous_state = nil
Utils.cached_pack_positions = {}
Utils.cached_consumable_positions = {}
Utils.cached_shop_card_positions = {}
Utils.cached_booster_positions = {}
Utils.pack_size = nil

Utils.gamestate_counter = 0


-- Session ID: Generated at game start
function Utils.generateSessionId()
    return tostring(os.time()) .. "_" .. tostring(math.random(1000, 9999))
end

-- Gamestate ID: Generated for each unique gamestate broadcast
function Utils.generateGamestateId()
    if not Utils.current_session_id then
        Utils.current_session_id = Utils.generateSessionId()
    end
    
    Utils.gamestate_counter = Utils.gamestate_counter + 1
    return Utils.current_session_id .. "_GS" .. Utils.gamestate_counter .. "_" .. os.time()
end 

function Utils.clearGamestateId()
    Utils.previous_state = Utils.current_gamestate_id
    Utils.current_gamestate_id = nil
end

function Utils.isPackState(state)
    return state == G.STATES.TAROT_PACK or
           state == G.STATES.PLANET_PACK or
           state == G.STATES.SPECTRAL_PACK or
           state == G.STATES.STANDARD_PACK or
           state == G.STATES.BUFFOON_PACK
end

function Utils.getPackTypeFromState(state)
    local pack_types = {
        [G.STATES.TAROT_PACK] = "tarot",
        [G.STATES.PLANET_PACK] = "planet",
        [G.STATES.SPECTRAL_PACK] = "spectral",
        [G.STATES.STANDARD_PACK] = "standard",
        [G.STATES.BUFFOON_PACK] = "buffoon"
    }
    return pack_types[state] or "unknown"
end

-- Initialize session on first call
function Utils.ensureSessionId()
    if not Utils.current_session_id then
        Utils.current_session_id = Utils.generateSessionId()
    end
    return Utils.current_session_id
end

-- Reset all IDs (for new game)
function Utils.resetAllIds()
    Utils.current_session_id = Utils.generateSessionId()
    sendDebugMessage("Reset all IDs, new session: " .. tostring(Utils.current_session_id))
end

-- Get current context IDs for ActionTracker
function Utils.getCurrentContextIds()
    return {
        session_id = Utils.current_session_id,
        round_id = Utils.current_round_id,
        blind_id = Utils.current_blind_id,
        shop_id = Utils.current_shop_id
    }
end

function Utils.cache_pack_positions()
    Utils.cached_pack_positions = {}
    
    if G and G.STATE and Utils.isPackState(G.STATE) then
        if G.pack_cards and G.pack_cards.cards then
            for i, card in ipairs(G.pack_cards.cards) do
                if card and card.config and card.config.center then
                    -- Store by direct card reference
                    Utils.cached_pack_positions[card] = i
                    
                    -- Also store by card key as backup
                    local card_key = card.config.center.key
                    if card_key then
                        Utils.cached_pack_positions[card_key .. "_" .. tostring(card)] = i
                    end
                end
            end
            sendDebugMessage("Cached pack positions for " .. #G.pack_cards.cards .. " cards")
        end
    end
end

function Utils.cache_consumable_positions()
    Utils.cached_consumable_positions = {}
    
    if G and G.consumeables and G.consumeables.cards then
        for i, card in ipairs(G.consumeables.cards) do
            if card and card.config and card.config.center then
                -- Store by direct card reference
                Utils.cached_consumable_positions[card] = i
                
                -- Also store by card key as backup
                local card_key = card.config.center.key
                if card_key then
                    Utils.cached_consumable_positions[card_key .. "_" .. tostring(card)] = i
                end
            end
        end
        sendDebugMessage("Cached consumable positions for " .. #G.consumeables.cards .. " cards")
    end
end

function Utils.cache_shop_card_positions()
    Utils.cached_shop_card_positions = {}
    
    if G and G.shop_jokers and G.shop_jokers.cards then
        for i, card in ipairs(G.shop_jokers.cards) do
            if card and card.config and card.config.center then
                -- Store by direct card reference
                Utils.cached_shop_card_positions[card] = i
                
                -- Also store by card key as backup
                local card_key = card.config.center.key
                if card_key then
                    Utils.cached_shop_card_positions[card_key .. "_" .. tostring(card)] = i
                end
            end
        end
        sendDebugMessage("Cached joker positions for " .. #G.shop_jokers.cards .. " cards")
    end
end

function Utils.cache_booster_positions()
    Utils.cached_booster_positions = {}
    
    if G and G.shop_booster and G.shop_booster.cards then
        for i, card in ipairs(G.shop_booster.cards) do
            if card and card.config and card.config.center then
                -- Store by direct card reference
                Utils.cached_booster_positions[card] = i
                
                -- Also store by card key as backup
                local card_key = card.config.center.key
                if card_key then
                    Utils.cached_booster_positions[card_key .. "_" .. tostring(card)] = i
                end
            end
        end
        sendDebugMessage("Cached booster positions for " .. #G.shop_booster.cards .. " cards")
    end
end

function Utils.getCardData(card)
    local _card = { }

    _card.label = card.label
    _card.name = card.config.card.name
    _card.suit = card.config.card.suit
    _card.value = card.config.card.value
    _card.card_key = card.config.card_key
    _card.chips = card:get_chip_bonus() or nil
    _card.mult = card:get_chip_mult() or nil
    _card.x_mult = card:get_chip_x_mult() or nil
    _card.x_h_mult = card:get_chip_h_x_mult() or nil
    _card.seal = card.seal and card.seal.key or nil
    _card.edition = card.edition and {
                            type = card.edition.type,
                            negative = card.edition.negative,
                            foil = card.edition.foil,
                            holo = card.edition.holo,
                            polychrome = card.edition.polychrome
                        } or nil

    if card.ability.set =='Joker' then
        _card.ability = card.ability
    end
    return _card
end

function Utils.get_cached_pack_position(card)
    if not card then return nil end
    
    -- Direct lookup by card reference
    local position = Utils.cached_pack_positions[card]
    if position then
        return position
    end
    
    -- Backup lookup by key + card reference
    local card_key = card.config and card.config.center and card.config.center.key
    if card_key then
        local backup_position = Utils.cached_pack_positions[card_key .. "_" .. tostring(card)]
        if backup_position then
            return backup_position
        end
    end
    
    return nil
end

function Utils.get_cached_consumable_position(card)
    if not card then return nil end
    
    -- Direct lookup by card reference
    local position = Utils.cached_consumable_positions[card]
    if position then
        return position
    end
    
    -- Backup lookup by key + card reference
    local card_key = card.config and card.config.center and card.config.center.key
    if card_key then
        local backup_position = Utils.cached_consumable_positions[card_key .. "_" .. tostring(card)]
        if backup_position then
            return backup_position
        end
    end
    
    return nil
end

function Utils.get_cached_shop_card_position(card)
    if not card then return nil end
    
    -- Direct lookup by card reference
    local position = Utils.cached_shop_card_positions[card]
    if position then
        return position
    end
    
    -- Backup lookup by key + card reference
    local card_key = card.config and card.config.center and card.config.center.key
    if card_key then
        local backup_position = Utils.cached_shop_card_positions[card_key .. "_" .. tostring(card)]
        if backup_position then
            return backup_position
        end
    end
    
    return nil
end

function Utils.get_cached_booster_position(card)
    if not card then return nil end
    
    -- Direct lookup by card reference
    local position = Utils.cached_booster_positions[card]
    if position then
        return position
    end
    
    -- Backup lookup by key + card reference
    local card_key = card.config and card.config.center and card.config.center.key
    if card_key then
        local backup_position = Utils.cached_booster_positions[card_key .. "_" .. tostring(card)]
        if backup_position then
            return backup_position
        end
    end
    
    return nil
end

function Utils.clear_pack_positions()
    Utils.cached_pack_positions = {}
    sendDebugMessage("Cleared pack position cache")
end

function Utils.clear_consumable_positions()
    Utils.cached_consumable_positions = {}
    sendDebugMessage("Cleared consumable position cache")
end

function Utils.clear_shop_card_positions()
    Utils.cached_shop_card_positions = {}
    sendDebugMessage("Cleared consumable position cache")
end

function Utils.clear_booster_positions()
    Utils.cached_booster_positions = {}
    sendDebugMessage("Cleared consumable position cache")
end

function Utils.getDeckData()
    local _deck = { }
    if G and G.deck and G.deck.cards then
        for i = 1, #G.deck.cards do
            local _card = Utils.getCardData(G.deck.cards[i])
            _deck[i] = _card
        end
    end
    return _deck
end

function Utils.getHandData()
    local _hand = { }
    if G and G.hand and G.hand.cards then
        if #G.hand.cards < 1 then
                return _hand
            end
        for i = 1, #G.hand.cards do
            local _card = Utils.getCardData(G.hand.cards[i])
            _hand[i] = _card
        end
    end

    return _hand
end

function Utils.getJokersData()
    local _jokers = { }

    if G and G.jokers and G.jokers.cards then
        for i = 1, #G.jokers.cards do
            local _card = Utils.getCardData(G.jokers.cards[i])
            _jokers[i] = _card
        end
    end

    return _jokers
end

function Utils.getConsumablesData()
    local _consumables = { }

    if G and G.consumables and G.consumables.cards then
        for i = 1, #G.consumeables.cards do
            local _card = Utils.getCardData(G.consumeables.cards[i])
            _consumables[i] = _card
        end
    end

    return _consumables
end

function Utils.getBlindData()
    local _blinds = {}

    if G and G.GAME then
        _blinds.ondeck = G.GAME.blind_on_deck
        
        -- Add current blind information if available (avoid circular refs)
        if G.GAME.blind then
            _blinds.current = {
                name = G.GAME.blind.name,
                key = G.GAME.blind.key,
                chips = G.GAME.blind.chips,
                mult = G.GAME.blind.mult
            }
        end
    end

    return _blinds
end

function Utils.getAnteData()
    local _ante = { }
    _ante.blinds = Utils.getBlindData()

    return _ante
end


function Utils.getShopData()
    local _shop = { }
    if not G or not G.shop then return _shop end
    
    _shop.reroll_cost = G.GAME.current_round.reroll_cost
    _shop.cards = { }
    _shop.boosters = { }
    _shop.vouchers = { }

    for i = 1, #G.shop_jokers.cards do
        _shop.cards[i] = Utils.getCardData(G.shop_jokers.cards[i])
    end

    for i = 1, #G.shop_booster.cards do
        _shop.boosters[i] = Utils.getCardData(G.shop_booster.cards[i])

    end

    for i = 1, #G.shop_vouchers.cards do
        _shop.vouchers[i] = Utils.getCardData(G.shop_vouchers.cards[i])
    end
    return _shop
end

-- Enhanced pack data for booster pack states
function Utils.getPackData()
    local _pack = {}
    
    if G and G.STATE and Utils.isPackState(G.STATE) then
        if G.pack_cards and G.pack_cards.cards then
            if #G.pack_cards.cards < 1 then
                return _pack
            end
            _pack.cards = {}
            for i, card in ipairs(G.pack_cards.cards) do
                if card and card.config and card.config.center then
                    _pack.cards[i] = {
                        key = card.config.center.key,
                        name = card.config.center.name,
                        set = card.ability and card.ability.set,
                        cost = card.cost or 0,
                        position = i,
                        -- Add edition/seal info if present
                        edition = card.edition and {
                            type = card.edition.type,
                            negative = card.edition.negative,
                            foil = card.edition.foil,
                            holo = card.edition.holo,
                            polychrome = card.edition.polychrome
                        } or nil,
                        seal = card.seal and card.seal.key or nil,
                        -- For playing cards, include rank/suit
                        rank = card.base and card.base.id or nil,
                        suit = card.base and card.base.suit or nil,
                        -- For jokers, include rarity
                        rarity = card.config and card.config.center and card.config.center.rarity or nil
                    }
                end
            end
            
            -- Add pack metadata
            _pack.info = {
                pack_type = Utils.getPackTypeFromState(G.STATE),
                total_cards = #G.pack_cards.cards,
                max_selections = G.GAME and G.GAME.pack_choices or 1,
                remaining_selections = G.GAME and G.GAME.pack_choices or 1
            }
        end
    end
    
    return _pack
end

function Utils.getHandScoreData()
    local _handscores = { }
    if G and G.GAME and G.GAME.blind then
        _handscores.score = G.GAME.chips
        _handscores.goal = G.GAME.blind.chips
    end

    return _handscores
end

function Utils.getTagsData()
    local _tags = { }

    return _tags
end

function Utils.getRoundData()
    local _current_round = { }

    if G and G.GAME and G.GAME.current_round then
        _current_round.discards_left = G.GAME.current_round.discards_left
    end

    return _current_round
end

function Utils.getHandLevelsData()
    local hand_levels = {}
    
    if G and G.GAME and G.GAME.hands then
        for hand_name, hand_data in pairs(G.GAME.hands) do
            if hand_data then
                hand_levels[hand_name] = {
                    level = hand_data.level or 0,
                    chips = hand_data.chips or 0,
                    mult = hand_data.mult or 0,
                    played = hand_data.played or 0,
                    visible = hand_data.visible or false
                }
            end
        end
    end
    
    return hand_levels
end

function Utils.getGameData()
    local _game = { }

    if G and G.STATE then
        _game.state = G.STATE
        _game.num_hands_played = G.GAME.hands_played
        _game.num_skips = G.GAME.Skips
        _game.round = G.GAME.round
        _game.discount_percent = G.GAME.discount_percent
        _game.interest_cap = G.GAME.interest_cap
        _game.inflation = G.GAME.inflation
        _game.dollars = G.GAME.dollars
        _game.max_jokers = G.GAME.max_jokers
        _game.bankrupt_at = G.GAME.bankrupt_at
        _game.chips = G.GAME.chips
        _game.hand_levels = Utils.getHandLevelsData()
    end

    return _game
end

function Utils.getGamestate()
    -- Ensure session ID exists
    Utils.ensureSessionId()
    
    
    local _gamestate = Utils.getGameData()
    
    -- Add all context IDs to gamestate
    _gamestate.session_id = Utils.current_session_id
    _gamestate.deck = Utils.getDeckData()
    _gamestate.hand = Utils.getHandData()
    _gamestate.jokers = Utils.getJokersData()
    _gamestate.consumables = Utils.getConsumablesData()
    _gamestate.ante = Utils.getAnteData()
    _gamestate.shop = Utils.getShopData()
    _gamestate.pack = Utils.getPackData()
    _gamestate.handscores = Utils.getHandScoreData()
    _gamestate.tags = Utils.getTagsData()
    _gamestate.current_round = Utils.getRoundData()

    return _gamestate
end


function Utils.parseaction(data)
    -- Protocol is ACTION|arg1|arg2
    action = data:match("^([%a%u_]*)")
    params = data:match("|(.*)")

    if action then
        local _action = Bot.ACTIONS[action]

        if not _action then
            return nil
        end

        local _actiontable = { }
        _actiontable[1] = _action

        if params then
            local _i = 2
            for _arg in params:gmatch("[%w%s,]+") do
                local _splitstring = { }
                local _j = 1
                for _str in _arg:gmatch('([^,]+)') do
                    _splitstring[_j] = tonumber(_str) or _str
                    _j = _j + 1
                end
                _actiontable[_i] = _splitstring
                _i = _i + 1
            end
        end

        return _actiontable
    end
end

Utils.ERROR = {
    NOERROR = 1,
    NUMPARAMS = 2,
    MSGFORMAT = 3,
    INVALIDACTION = 4,
}

function Utils.validateAction(action)
    if action and #action > 1 and #action > Bot.ACTIONPARAMS[action[1]].num_args then
        return Utils.ERROR.NUMPARAMS
    elseif not action then
        return Utils.ERROR.MSGFORMAT
    else
        if not Bot.ACTIONPARAMS[action[1]].isvalid(action) then
            return Utils.ERROR.INVALIDACTION
        end
    end

    return Utils.ERROR.NOERROR
end

function Utils.isTableUnique(table)
    if table == nil then return true end

    local _seen = { }
    for i = 1, #table do
        if _seen[table[i]] then return false end
        _seen[table[i]] = table[i]
    end

    return true
end

function Utils.isTableInRange(table, min, max)
    if table == nil then return true end

    for i = 1, #table do
        if table[i] < min or table[i] > max then return false end
    end
    return true
end

return Utils