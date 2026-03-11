#!/usr/bin/perl
use strict;
use warnings;

# Subroutine to calculate time differences between start and finish times
#
# The subroutine can be used in two ways:
# As a module: Import it and call calculate_time_differences(@lines)
# As a standalone script: Run it from the command line with input from STDIN or files as arguments
#
# Input:   Array of lines in format: "node_name\tstart_time\tfinish_time"
#          where times are in HH:MM:SS format
# Returns: Hash reference with node names as keys and time differences in seconds as values
#          Also includes formatted time differences as HH:MM:SS in a separate hash
#
sub calculate_time_differences {
    my @lines = @_;
    my %results;
    my %formatted_results;
    
    foreach my $line (@lines) {
        chomp $line;
        next if $line =~ /^\s*$/;  # Skip empty lines
        
        # Parse the line: node_name, start_time, finish_time
        # Handle both tab and space separators
        my ($node, $start_time, $finish_time) = split /\s+/, $line, 3;
        
        next unless defined $node && defined $start_time && defined $finish_time;
        
        # Convert times to seconds
        my $start_seconds = time_to_seconds($start_time);
        my $finish_seconds = time_to_seconds($finish_time);
        
        # Check if conversion was successful
        next if $start_seconds < 0 || $finish_seconds < 0;
        
        # Calculate difference
        my $diff_seconds = $finish_seconds - $start_seconds;
        
        # Handle case where finish time is on the next day (crosses midnight)
        if ($diff_seconds < 0) {
            $diff_seconds += 86400;  # Add 24 hours in seconds
        }
        
        $results{$node} = $diff_seconds;
        $formatted_results{$node} = seconds_to_time($diff_seconds);
    }
    
    return {
        seconds => \%results,
        formatted => \%formatted_results
    };
}

# Helper subroutine to convert HH:MM:SS to seconds
sub time_to_seconds {
    my $time_str = shift;
    return -1 unless defined $time_str;
    
    # Match HH:MM:SS format
    if ($time_str =~ /^(\d{1,2}):(\d{2}):(\d{2})$/) {
        my ($hours, $minutes, $seconds) = ($1, $2, $3);
        return $hours * 3600 + $minutes * 60 + $seconds;
    }
    
    return -1;  # Invalid format
}

# Helper subroutine to convert seconds to HH:MM:SS format
sub seconds_to_time {
    my $total_seconds = shift;
    my $hours = int($total_seconds / 3600);
    my $minutes = int(($total_seconds % 3600) / 60);
    my $seconds = $total_seconds % 60;
    
    return sprintf("%02d:%02d:%02d", $hours, $minutes, $seconds);
}

# Example usage:
if (!caller()) {
    # Read from STDIN or command line arguments
    my @input_lines;
    if (@ARGV) {
        # Read from files specified as arguments
        foreach my $file (@ARGV) {
            open my $fh, '<', $file or die "Cannot open $file: $!";
            push @input_lines, <$fh>;
            close $fh;
        }
    } else {
        # Read from STDIN
        @input_lines = <STDIN>;
    }
    
    my $results = calculate_time_differences(@input_lines);
    
    # Print results
    foreach my $node (sort keys %{$results->{seconds}}) {
        my $diff_seconds = $results->{seconds}->{$node};
        my $diff_formatted = $results->{formatted}->{$node};
        print "$node\t$diff_formatted\t$diff_seconds seconds\n";
    }
}

1;
