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
    
    if ($line =~ /^function\s+_tradelog/) {
        $skip = 1;
        $depth = 1;
        next;
    }
    
    push @output, $line;
}

open $fh, '>', $file or die "Cannot write $file: $!";
print $fh @output;
close $fh;

print "Removed all _tradelog* functions\n";
