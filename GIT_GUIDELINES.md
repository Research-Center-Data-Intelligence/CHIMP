## Branching Strategy

Voor dit project hanteren we een branching strategy die aansluit bij open source development:

- **Main branch:** De `main` branch is de basis branch van de repository. Hier staat altijd de meest stabiele en actuele versie van de code.
- **Feature branches:** Medewerkers werken aan nieuwe features, bugfixes of verbeteringen in een eigen feature branch. Deze branches worden aangemaakt vanaf de `main` branch en na afronding via een pull request samengevoegd.
- **Forks voor studenten:** Studenten maken een fork van de repository in hun eigen GitHub omgeving. Zij werken in hun eigen fork en kunnen via pull requests bijdragen aan het hoofdproject.

Deze aanpak zorgt voor een duidelijke scheiding tussen stabiele code, actieve ontwikkeling door medewerkers en externe bijdragen door studenten.

## Voorkom lang openstaande branches

Het is belangrijk om wijzigingen in een feature branch of fork tijdig terug te brengen naar de `main` branch via een pull request. Lang openstaande branches met veel wijzigingen die niet in `main` terechtkomen, vergroten de kans op merge-conflicten en maken het lastiger om nieuwe functionaliteit te integreren. Door regelmatig een pull request aan te maken en je werk samen te voegen met `main`:

- Blijft de codebase overzichtelijk en up-to-date.
- Voorkom je dat je werk veroudert of moeilijk te integreren wordt.
- Draag je sneller bij aan het gezamenlijke projectresultaat.

Kortom: houd branches kort en merge ze tijdig om een gezonde en samenwerkende ontwikkelomgeving te behouden.

## Sync regelmatig met main?

Het is essentieel om je feature branch of fork regelmatig te synchroniseren met de `main` branch. Hierdoor voorkom je dat je achterloopt op recente wijzigingen, bugfixes of nieuwe features die door anderen zijn toegevoegd. Door frequent te syncen:

- Minimaliseer je merge-conflicten bij het samenvoegen van je werk.
- Zorg je dat je ontwikkelt op basis van de meest actuele en stabiele code.
- Kun je sneller inspelen op veranderingen in het project.

## Hoe sync je met main?
Hieronder volgend de commandline opties, het is ook mogelijk dit te doen in de VSCode GUI.

### In een feature branch (zelfde repository)
1. Zorg dat je lokale repository up-to-date is:
	```
	git fetch origin
	```
2. Checkout je feature branch:
	```
	git checkout <jouw-feature-branch>
	```
3. Merge de laatste wijzigingen van main:
	```
	git merge origin/main
	```
4. Los eventuele merge-conflicten op, commit en push je branch indien nodig.

### In een fork (eigen repository)
1. Voeg het originele project toe als remote (indien nog niet gedaan):
	```
	git remote add upstream https://github.com/Research-Center-Data-Intelligence/CHIMP.git
	```
2. Haal de laatste wijzigingen van het hoofdproject op:
	```
	git fetch upstream
	```
3. Checkout je eigen branch:
	```
	git checkout <jouw-branch>
	```
4. Merge de wijzigingen van main uit het hoofdproject:
	```
	git merge upstream/main
	```
5. Los eventuele merge-conflicten op, commit en push je branch indien nodig.


## Pull request guidelines

TODO: aanvullen door Bryan.



