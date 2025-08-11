
Utils = { }

Utils.current_session_id = nil
Utils.current_round_id = nil
Utils.current_blind_id = nil
Utils.current_shop_id = nil
Utils.previous_state = nil


-- Session ID: Generated at game start
function Utils.generateSessionId()
    return tostring(os.time()) .. "_" .. tostring(math.random(1000, 9999))
end

function Utils.generateRoundId()
    if not Utils.current_session_id then
        Utils.current_session_id = Utils.generateSessionId()
    end
    
    local round_num = G.GAME and G.GAME.round or 0
    local ante_num = G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 0
    return Utils.current_session_id .. "_R" .. ante_num .. "-" .. round_num .. "_" .. os.time()
end

-- Blind ID: Generated when entering blind selection or blind context states
function Utils.generateBlindId()
    if not Utils.current_session_id then
        Utils.current_session_id = Utils.generateSessionId()
    end
    
    local round_num = G.GAME and G.GAME.round or 0
    local ante_num = G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 0
    return Utils.current_session_id .. "_B" .. ante_num .. "-" .. round_num .. "_" .. os.time()
end

-- Shop ID: Generated when entering SHOP state  
function Utils.generateShopId()
    if not Utils.current_session_id then
        Utils.current_session_id = Utils.generateSessionId()
    end
    
    local round_num = G.GAME and G.GAME.round or 0
    local ante_num = G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 0
    return Utils.current_session_id .. "_S" .. ante_num .. "-" .. round_num .. "_" .. os.time()
end

-- legacy id generators
function Utils.generateBlindSelectionId()
    if G and G.GAME and G.STATE == G.STATES.BLIND_SELECT then
        local blind_id_parts = {
            tostring(G.GAME.round or 0),
            tostring(G.GAME.round_resets.ante or 0),
            tostring(os.time()) -- Add timestamp for uniqueness
        }
        
        -- Add blind choice keys if available (avoid complex objects)
        if G.GAME.round_resets and G.GAME.round_resets.blind_choices then
            for i, choice in ipairs(G.GAME.round_resets.blind_choices) do
                if choice and choice.key then
                    table.insert(blind_id_parts, tostring(choice.key))
                end
            end
        end
        
        return table.concat(blind_id_parts, "_")
    end
    return nil
end

-- State context helpers
function Utils.isBlindContextState(state)
    return state == G.STATES.BLIND_SELECT or
           state == G.STATES.SELECTING_HAND or
           state == G.STATES.HAND_PLAYED or
           state == G.STATES.DRAW_TO_HAND or
           state == G.STATES.ROUND_EVAL
end

function Utils.isShopContextState(state)
    return state == G.STATES.SHOP or
           state == G.STATES.TAROT_PACK or
           state == G.STATES.PLANET_PACK or
           state == G.STATES.SPECTRAL_PACK or
           state == G.STATES.STANDARD_PACK or
           state == G.STATES.BUFFOON_PACK or
           state == G.STATES.PLAY_TAROT
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

-- ID management based on state transitions
function Utils.updateContextIds(current_state, previous_state)
    -- Generate new IDs when entering context states
    if current_state == G.STATES.NEW_ROUND and (not previous_state or previous_state ~= G.STATES.NEW_ROUND) then
        Utils.current_round_id = Utils.generateRoundId()
    end
    
    if current_state == G.STATES.BLIND_SELECT and (not previous_state or previous_state ~= G.STATES.BLIND_SELECT) then
        Utils.current_blind_id = Utils.generateBlindId()
    end
    
    if current_state == G.STATES.SHOP and (not previous_state or previous_state ~= G.STATES.SHOP) then
        Utils.current_shop_id = Utils.generateShopId()
    end
    
    -- Clear context IDs when leaving contexts
    if previous_state and Utils.isBlindContextState(previous_state) and not Utils.isBlindContextState(current_state) then
        -- Keep blind_id for historical reference, don't clear it
    end
    
    if previous_state and Utils.isShopContextState(previous_state) and not Utils.isShopContextState(current_state) then
        -- Keep shop_id for historical reference, don't clear it
    end
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
    Utils.current_round_id = nil
    Utils.current_blind_id = nil
    Utils.current_shop_id = nil
    Utils.previous_state = nil
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

function Utils.getCardData(card)
    local _card = { }

    _card.label = card.label
    _card.name = card.config.card.name
    _card.suit = card.config.card.suit
    _card.value = card.config.card.value
    _card.card_key = card.config.card_key

    return _card
end

function Utils.getDeckData()
    local _deck = { }

    return _deck
end

function Utils.getHandData()
    local _hand = { }

    if G and G.hand and G.hand.cards then
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
        
        -- IDs - ensure we have proper blind ID for blind context states
        if Utils.isBlindContextState(G.STATE) then
            if not Utils.current_blind_id then
                Utils.current_blind_id = Utils.generateBlindId()
                sendDebugMessage("Generated blind ID in getBlindData: " .. tostring(Utils.current_blind_id))
            end
            _blinds.blind_id = Utils.current_blind_id
        end
        
        -- Legacy selection ID for backward compatibility
        if G.STATE == G.STATES.BLIND_SELECT then
            _blinds.selection_id = Utils.generateBlindSelectionId()
        end
    end

    return _blinds
end

function Utils.getAnteData()
    local _ante = { }
    _ante.blinds = Utils.getBlindData()

    return _ante
end

function Utils.getBackData()
    local _back = { }

    return _back
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

    -- Ensure we have shop ID for shop context states
    if Utils.isShopContextState(G.STATE) then
        if not Utils.current_shop_id then
            Utils.current_shop_id = Utils.generateShopId()
            sendDebugMessage("Generated shop ID in getShopData: " .. tostring(Utils.current_shop_id))
        end
        _shop.shop_id = Utils.current_shop_id
    end

    return _shop
end

-- Enhanced pack data for booster pack states
function Utils.getPackData()
    local _pack = {}
    
    if G and G.STATE and Utils.isPackState(G.STATE) then
        if G.pack_cards and G.pack_cards.cards then
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
    end

    return _game
end

function Utils.getGamestate()
    -- Ensure session ID exists
    Utils.ensureSessionId()
    
    -- Update context IDs based on current state
    if G and G.STATE then
        Utils.updateContextIds(G.STATE, Utils.previous_state)
    end
    
    local _gamestate = Utils.getGameData()
    
    -- Add all context IDs to gamestate
    _gamestate.session_id = Utils.current_session_id
    _gamestate.round_id = Utils.current_round_id
    
    -- Add context-specific IDs based on current state
    if G and G.STATE then
        if Utils.isBlindContextState(G.STATE) then
            if not Utils.current_blind_id then
                Utils.current_blind_id = Utils.generateBlindId()
                sendDebugMessage("Generated blind ID in getGamestate: " .. tostring(Utils.current_blind_id))
            end
            _gamestate.blind_id = Utils.current_blind_id
            _gamestate.context_id = Utils.current_blind_id
            _gamestate.context_type = "blind_session"
        elseif Utils.isShopContextState(G.STATE) then
            if not Utils.current_shop_id then
                Utils.current_shop_id = Utils.generateShopId()
                sendDebugMessage("Generated shop ID in getGamestate: " .. tostring(Utils.current_shop_id))
            end
            _gamestate.shop_id = Utils.current_shop_id
            _gamestate.context_id = Utils.current_shop_id
            _gamestate.context_type = "shop_session"
        end
    end
    
    _gamestate.deckback = Utils.getBackData()
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