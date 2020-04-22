// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "MLR_BP2GameMode.h"
#include "MLR_BP2Character.h"
#include "UObject/ConstructorHelpers.h"

AMLR_BP2GameMode::AMLR_BP2GameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
