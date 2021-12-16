class ChatHelper {

    constructor() { }

    //public functions
    initializeClientChatObject(clientUniqueName, clientName) {
        let clientChatObject;
        return new Promise(resolve => {
            getNewTokenService(clientUniqueName).then((data) => {
                this.createClientChatObject(data.token).then((client) => {
                    clientChatObject = client;

                    // when the access token is about to expire, refresh it
                    clientChatObject.on('tokenAboutToExpire', () => { this.refreshToken(clientChatObject, clientUniqueName); });

                    // if the access token already expired, refresh it
                    clientChatObject.on('tokenExpired', () => { this.refreshToken(clientChatObject, clientUniqueName); });
                    resolve({ status: true, clientChatObject: clientChatObject, message: 'Initialization complete successfully for "' + clientName + '"', code: 200 });
                }).catch(error => {
                    resolve({ status: false, clientChatObject: null, message: error.message, code: error.code });
                });
            });

        });
    }
    refreshToken(clientChatObject, clientIdentity) {

        getNewTokenService(clientIdentity).then((data) => {
            this.updateToken(clientChatObject, data.token);
        }).catch((error) => {
            console.error(error);
        });
    }
    createClientChatObject(token) {
        return Twilio.Chat.Client.create(token);
    }
    updateToken(clientChatObject, token) {
        clientChatObject.updateToken(token);
    }
    getChannelByUniqueName(clientChatObject, uniqueName) {
        return new Promise(resolve => {
            clientChatObject.getChannelByUniqueName(uniqueName).then(result => {
                resolve({ status: true, channel: result, message: 'success', code: 200 });
            }).catch(error => {

                resolve({ status: false, channel: null, message: error.message, code: error.body ? error.body.code : -1 });
            });
        });
    }
    eventHasBeenRegisteredOnObjectBefore(objectInstance, eventName) {
        return Object.keys(objectInstance._events).findIndex(a => a == eventName) > -1;
    }
    checkChannelExistence(clientChatObject, uniqueName) {

        return this.getChannelByUniqueName(clientChatObject, uniqueName);
    }
    createNewChannel(clientChatObject, uniqueName, friendlyName, isPrivate) {
        return new Promise((resolve) => {
            clientChatObject.createChannel({
                uniqueName: uniqueName,
                friendlyName: friendlyName,
                isPrivate: isPrivate
            }).then(channel => {
                resolve({ status: true, channel: channel, message: 'channel created', code: 200 });
            }).catch(error => {
                resolve({ status: false, channel: null, message: error.message, code: error.code });
            });
        });
    }
    getChannelMembers(channel) {
        return new Promise(resolve => {
            let members = [];
            for (const [key, value] of channel.members._c.entries())
                members.push(value);
            resolve(members);
        });
    }
    updateUserAttributes(clientChatObject, attributes) {
        return new Promise((resolve) => {
            clientChatObject.user.updateAttributes(attributes).then(result => {
                resolve({ status: true, code: 200, message: "success" });
            }).catch(error => {
                resolve({ status: false, code: error.body.code, message: error.message });
            });
        });
    }
    getChannelUsersDetails(channel) {
        return new Promise(resolve => {

            channel.getUserDescriptors().then(userDescriptors => {
                resolve({ status: true, userDetails: userDescriptors.items, message: error.message });
            }).catch(error => {
                resolve({ status: false, userDetails: null, message: error.message });
            });
        });
    }
    deleteChannel(channel) {
        return new Promise((resolve) => {
            channel.delete().then(data => {
                resolve({ status: true, message: 'channel deleted successfully' });
            }).catch((error) => {
                resolve({ status: false, message: error.message });
            });
        })

    }
    getUserDescriptor(clientChatObject, clientIdentity) {
        return clientChatObject.getUserDescriptor(clientIdentity);
    }
    getUser(clientChatObject, clientIdentity) {
        return clientChatObject.getUser(clientIdentity);
    }
    joinToChannel(channel) {
        return new Promise((resolve) => {
            channel.join().then(data => {
                channel.removeAllListeners();
                resolve({ status: true, message: 'Client joined to channel', code: 200 });
            }).catch((error) => {
                channel.removeAllListeners();
                resolve({ status: false, code: error.code, message: error.message, code: error.code });
            });
        });

    }
    leaveChannel(channel) {
        return new Promise(resolve => {
            channel.leave().then(channel => {
                channel.removeAllListeners();
                resolve({ status: true, message: 'Client left channel.' });
            }).catch(error => {
                resolve({ status: false, message: error.message });
            });
        });
    }
    removeMember(channel, userIdentity) {
        return new Promise(resolve => {
            channel.removeMember(userIdentity).then(result => {
                resolve({ status: true, message: 'Member removed.' });
            }).catch(error => {
                resolve({ status: false, message: error.message });
            });
        });
    }
    addMember(channel, userIdentity) {
        return new Promise(resolve => {
            channel.add(userIdentity).then(result => {
                resolve({ status: true, message: 'Member removed.' });
            }).catch(error => {
                resolve({ status: false, message: error.message });
            });
        });

    }
    sendMessage(channel, message, additionalProperties = null) {
        return channel.sendMessage(message, additionalProperties ? additionalProperties : null);
    }
    serverSideDeleteChannel(channelId, clientUniqueName) {
        return new Promise((resolve) => {
            deleteChannelService(channelId, clientUniqueName).then(data => {
                resolve({ status: data.status, message: data.message, code: data.code });
            }).catch((error) => {
                resolve({ status: false, message: error.message, code: error.code ? error.code : undefined });
            });

        });
    }
    removeAllChannelListeners(channel) {
        channel.removeAllListeners();
    }
    getAllFormattedMessages(channel, clientUniqueName) {
        return new Promise((resolve) => {
            getAllFormattedMessagesService(channel.sid, clientUniqueName).then(result => {
                resolve({ status: true, message: "", data: result.d });
            }).catch(error => {
                resolve({ status: true, message: error.message });
            });
        });
    }
    getAllRawMessages(channel, clientUniqueName) {
        return new Promise((resolve) => {
            return getAllRawMessagesService(channel.sid, clientUniqueName).then(result => {
                resolve({ status: true, message: "data fetched successfully", data: result.d });
            }).catch(error => {
                resolve({ status: true, message: error.message });
            });
        });
    }
    checkMemberExistenceOnChannel(channel, clientUniqueName) {
        return new Promise(resolve => {
            channel.getMemberByIdentity(clientUniqueName).then(result => {
                resolve({ status: true, message: 'User is member of channel' });
            }).catch(error => {
                resolve({ status: false, message: error.message });
            });
        });

    }
    sendTheTypingIndicatorSignal(channel) {
        channel.typing();
    }

    //events
    addListenerOnTypingEndedEvent(channel, externalFunction) {

        if (this.eventHasBeenRegisteredOnObjectBefore(channel, 'typingEnded'))
            return;

        channel.on('typingEnded', data => {

            if (externalFunction)
                externalFunction();
        });


    }
    addListenerOnTypingStartedEvent(channel, externalFunction) {

        if (this.eventHasBeenRegisteredOnObjectBefore(channel, 'typingStarted'))
            return;

        channel.on('typingStarted', data => {
            if (externalFunction)
                externalFunction();

        });


    }
    addListenerOnChannelAddedEvent(chatClientObject, externalFunction) {

        if (this.eventHasBeenRegisteredOnObjectBefore(chatClientObject, 'channelAdded'))
            return;

        chatClientObject.on('channelAdded', data => {
            if (externalFunction)
                externalFunction(data);
        });


    }
    addListenerOnChannelRemovedEvent(chatClientObject, externalFunction) {

        if (this.eventHasBeenRegisteredOnObjectBefore(chatClientObject, 'channelRemoved'))
            return;

        chatClientObject.on('channelRemoved', data => {
            if (externalFunction)
                externalFunction(data);
        });

    }
    addListenerOnChannelInvitedEvent(chatClientObject, externalFunction) {

        if (this.eventHasBeenRegisteredOnObjectBefore(chatClientObject, 'channelInvited'))
            return;

        chatClientObject.on('channelInvited', invitation => {
            if (externalFunction)
                externalFunction();
        });
    }
    addListenerOnMemberJoinedEvent(channel, externalFunction) {


        if (this.eventHasBeenRegisteredOnObjectBefore(channel, 'memberJoined'))
            return;
        //change test
        channel.on('memberJoined', member => {

            if (externalFunction)
                externalFunction(member);



        });


    }
    addListenerOnMemberLeftEvent(channel, externalFunction) {

        if (this.eventHasBeenRegisteredOnObjectBefore(channel, 'memberLeft'))
            return;

        channel.on('memberLeft', member => {

            if (externalFunction)
                externalFunction(member);
        });

    }
    addListenerOnMessageAddedEvent(channel, externalFunction) {


        if (this.eventHasBeenRegisteredOnObjectBefore(channel, 'messageAdded'))
            return;
        //this method should be revised
        channel.on('messageAdded', message => {

            if (externalFunction)
                externalFunction(message);
        });

    }


}

