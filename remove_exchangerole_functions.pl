#!/usr/bin/perl
use strict;
use warnings;

my $file = "Xch/src/XchCore.jl";
open my $fh, '<', $file or die "Cannot open $file: $!";
my @lines = <$fh>;
close $fh;

my @output;
my $skip = 0;
my $depth = 0;

# Functions to remove that contain ExchangeRole
my %remove_functions = (
    '_ensurewschannel!' => 1,
    '_adapterwsdfsnapshot' => 1,
    '_adapterwsheartbeat' => 1,
    '_routedbc' => 1,
    '_routedModule' => 1,
    'setrole!' => 1,
);

for my $i (0..$#lines) {
    my $line = $lines[$i];
    
    if ($skip) {
        if ($line =~ /^end\s*$/) {
            $depth--;
            $skip = 0 if $depth == 0;
        } elsif ($line =~ /^function\s+/) {
            $depth++;
        }
        next;
    }
    
    # Check if this is a function to remove
    my $should_remove = 0;
    foreach my $fname (keys %remove_functions) {
        if ($line =~ /^function\s+$fname[\(\s]/) {
            $should_remove = 1;
            last;
        }
    }
    
    if ($should_remove) {
        $skip = 1;
        $depth = 1;
        next;
    }
    
    push @output, $line;
}

open $fh, '>', $file or die "Cannot write $file: $!";
print $fh @output;
close $fh;

print "Removed all functions with ExchangeRole parameters\n";
