#!/bin/bash

# The purpose of this script is to recursively count the number of lines in a
# directory and return that number to the user.
#
# Walter Blaurock 
# May 2, 2011

# provide some help
if [[ $# = 1 && $1 = "--help" ]] || [[ $# = 1 && $1 = "-h" ]];
then

	echo "Returns the number of lines of text found in the (current) directory."
	echo "usage: lines [path]"
	exit 1

# if an argument is given, check if its a directory, and cd into it
elif [ $# = 1 ];
then
	cd $1 > /dev/null 2>&1
	# if cd returns an error, so should we
	if [ $? != 0 ]; then
		echo "Argument is not a valid directory."
		exit 1
	fi
fi

# execute the actual command in the current directory
line_count=`find . -type f -exec wc -l {} \; | awk '{total += $1} END{print total }'`

dir=$1
if [[ $dir = "" ]];
then
	dir="."
fi

echo "Found $line_count lines of text in $dir"

exit 0