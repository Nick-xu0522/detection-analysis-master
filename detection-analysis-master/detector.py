# encoding: utf-8
import relations
import numpy as np
import helper
import math
import os
from graphviz import Digraph



def transform(traces, types):
    """
    transform traces into relation matrix
    """
    n_traces = len(traces)
    matrixs = {}
    for relation_type in types:
        matrix = {}
        for i, trace in enumerate(traces):
            for key, value in relation_type['extract'](trace).items():
                array = matrix.setdefault(key, np.full(n_traces, relation_type['default'], np.int8))
                array[i] = value
        matrixs[relation_type['name']] = matrix
    return matrixs


def get_candidates(matrixs, min_window_size):
    """
    Getting candidates from relation matrix.
    """
    candidates = []
    for _, matrix in matrixs.items():
        for _, stream in matrix.items():
            candidates.extend(partition(stream, min_window_size))
    print("Candidates:",candidates)
    print("--------------------------------------------------------------------------")
    return candidates



def combine(candidates, radius, bounds, min_window_size, alpha=0.9):
    """
    combine
    """
    clusters = helper.DBSCAN_1d(candidates, radius, 1)
    clusters.sort(key=lambda x: -len(x))
    print("Clusters:", clusters)
    print("----------------------------------------------------------------------------------")
    partitioned = [bounds[0], bounds[-1]]
    for cluster in clusters:
        center = sum(cluster) / len(cluster)
        for i, value in enumerate(partitioned):
            if value == center:
                break
            if value > center:
                if partitioned[i] - partitioned[i - 1] >= 2 * min_window_size:
                    left = center - partitioned[i - 1]
                    right = partitioned[i] - center
                    if min_window_size * alpha <= left <= min_window_size:
                        partitioned.insert(i, partitioned[i - 1] + min_window_size)
                    elif min_window_size * alpha <= right <= min_window_size:
                        partitioned.insert(i, partitioned[i] - min_window_size)
                    elif left >= min_window_size and right >= min_window_size:
                        partitioned.insert(i, center)
    return partitioned


def detect(traces, min_window_size, radius, relation_types):
    """
    detect
    """
    matrixs = transform(traces, relation_types)
    candidates = get_candidates(matrixs, min_window_size)
    change_points = combine(candidates, radius, [0, len(traces)], min_window_size)
    return change_points


def partition(stream, min_window_size):
    """
    Detecting candiate points from stream.
    """
    candidates = set()
    begin = -1
    count = 0
    for index, value in enumerate(stream):
        if begin == -1 or stream[begin] != value:
            if count >= min_window_size:
                candidates.add(begin)
                candidates.add(index)
            begin, count = index, 0
        count += 1
    if count >= min_window_size:
        candidates.add(begin)
    return candidates



def test():
    log_dir = 'C:\\Users\\Administrator\\Desktop\\cf2.5k.mxml'
    for filename in os.listdir(log_dir):
        path = os.path.join(log_dir, filename)
        if os.path.isfile(path):
            traces = helper.parse_mxml(path)
            change_points = detect(traces, 100, 10, [relations.CONTEXT_RALATION])
            print("Change points detected: ", change_points)
            print("----------------------------------------------------------------------------------")
# calculation time

# Get the node
def get_next_code(dic):
    next_code = 'a'
    for k, v in dic.items():
        if next_code < v:
            next_code = v
    next_code = chr(ord(next_code) + 1)
    return next_code


def draw_edges(node1, node2, gz_list, i):
    hou_index_1 = 0
    hou_index_2 = 0
    flag1 = True
    flag2 = True
    gz = gz_list[0][i]
    dic = gz_list[1][i]
    while hou_index_1 < len(node1[1][0]) and hou_index_2 < len(node2[1][0]):
        if not flag1:
            # If hou_index_1 is out of bounds, hou_index_2 line is connected to 1
            for i_index in range(hou_index_2, len(node2[1][0])):
                key = node2[1][0][i_index]
                if key not in dic:
                    # Add nodes to the i Figure
                    next_code = get_next_code(dic)
                    gz.node(next_code, key, style='filled', color='blue')
                    dic[key] = next_code
                gz.edge(dic[node1[0]], dic[key], color='yellow')
            return
        if not flag2:
            # If hou_index_2 is out of bounds, hou_index_1 the cable is connected and marked with a red color
            for i_index in range(hou_index_1, len(node1[1][0])):
                key = node1[1][0][i_index]
                if key not in dic:
                    next_code = get_next_code(dic)
                    gz.node(next_code, key, style='filled', color='blue')
                    dic[key] = next_code
                gz.edge(dic[node1[0]], dic[key], color='green')
            return
        hou_node_1 = node1[1][0][hou_index_1]
        hou_node_2 = node2[1][0][hou_index_2]
        if hou_node_1 == hou_node_2:
            hou_index_1 += 1
            hou_index_2 += 1
            pass
        elif hou_node_1 > hou_node_2:
            # The node name is not the same, the front big, the back plus 1, the back output
            # print('hou_node_1 =', hou_node_1)
            # print('hou_node_2 =', hou_node_2)
            if hou_node_2 not in dic:
                # If the node does not exist, the node is created and wired
                print("The node not in dic")
                next_code = get_next_code(dic)
                gz.node(next_code, hou_node_2, style='filled', color='blue')
                dic[hou_node_2] = next_code
            gz.edge(dic[node1[0]], dic[hou_node_2], color='blue')
            hou_index_2 += 1
        else:
            # hou_node_1 < hou_node_2:
            # The node name is not the same, the back is large, the front plus 1, the previous output, and marked on the figure
            gz.edge(dic[node1[0]], dic[hou_node_1], color='blue')
            hou_index_1 += 1
        # Detect if it is out of bounds
        if hou_index_1 == len(node1[1][0]) and hou_index_2 != len(node2[1][0]):
            flag1 = False
            hou_index_1 = hou_index_1 - 1
        if hou_index_2 == len(node2[1][0]) and hou_index_1 != len(node1[1][0]):
            flag2 = False
            hou_index_2 = hou_index_2 - 1

def transform(traces, types):
    """
    transform traces into relation matrix
    """
    n_traces = len(traces)
    matrixs = {}
    for relation_type in types:
        matrix = {}
        for i, trace in enumerate(traces):
            for key, value in relation_type['extract'](trace).items():
                array = matrix.setdefault(key, np.full(n_traces, relation_type['default'], np.int8))
                array[i] = value
        matrixs[relation_type['name']] = matrix
    return matrixs

def main():
        traces = helper.parse_mxml("C://Users//Administrator//Desktop//cf2.5k.mxml")
        minimum_window_size = 200
        radius = 10
        change_points = detect(traces, minimum_window_size, radius, [relations.CONTEXT_RALATION])
        list_graph_dict = []
        nodes_number = 0
        # gz_list to store multiple graph objects, as well as a dictionary: the key word ---> the graph
        gz_list = [], []
        ms = []
        for i in range(1,len(change_points)-1):
            print("------------------------------------------------------------------------------")
            print("Change points detected: ", math.ceil(change_points[i]))
            lis = traces[math.ceil(change_points[i]):math.ceil(change_points[i + 1])]
            nodes = set()
            for li in lis:
                for l in li:
                    nodes.add(l)
            nodes_number = max(len(nodes), nodes_number)
            gz = Digraph("流程图"+str(i), 'comment', None, None, 'jpg', None, "UTF-8",
                         {'rankdir': 'TB'},
                         {'color': 'black', 'fontcolor': 'black', 'fontname': 'Time New Roman', 'fontsize': '12',
                          'style': 'rounded', 'shape': 'box'},
                         {'color': '#999999', 'fontcolor': '#888888', 'fontsize': '10', 'fontname': 'FangSong'}, None,
                         False)
            dict = {}
            dict_name = {}
            # edge_dict is used to access the front-driven successor
            edge_dict = {}
            for i, j in enumerate(nodes):
                # Create nodes
                gz.node("%c" % (i + 97), j)
                # Record the relationship between the node name and the code name, which is the dictionary keyword and the code name is the value
                dict[j] = "%c" % (i + 97)
                # Record the relationship between the node name and the code name, which is the keyword of the dictionary and the name is the value
                dict_name["%c" % (i + 97)] = j
                # Each keyword has an array of two elements, the first save followed by the second save forward
                edge_dict[j] = [], []

            # Traverses all the connections and saves them to the set collection
            edges = set()
            for li in lis:
                for l in range(len(li) - 1):
                    s1 = dict[li[l]]
                    s2 = dict[li[l + 1]]
                    edges.add(s1 + s2)
            # Create edges on the graph, depending on the set collection
            gz.edges(edges)
            gz_list[0].append(gz)
            gz_list[1].append(dict)

            # Traverses all the edge collections and saves the successor edge_dict dictionary
            for edge in edges:
                # The successor is saved to the 0 position
                edge_dict[dict_name[edge[0]]][0].append(dict_name[edge[1]])
                # The front drive is saved to the 1 position
                edge_dict[dict_name[edge[1]]][1].append(dict_name[edge[0]])
            for (k,v) in edge_dict.items():
                v[0].sort()
                v[1].sort()
            # The dictionary information for the current figure is added list_graph_dict
            list_graph_dict.append(edge_dict)
            Sub_matrix = relations.transform(lis, [relations.CONTEXT_RALATION])
            ms.append(Sub_matrix)
        sum_gx = set()
        a_gx = []
        # Adjacent matrices compare columns with different output activities
        for i in range(len(ms)):
            a = 0
            for (k, v) in ms[i].items():
                sum_gx.add(k)
                if i is not len(ms) - 1 and k in ms[i + 1]:
                    a += 1
                    v_ = ms[i + 1][k]
                    # print(v_)
                    num_inter = min(len(v), len(v_))
                    list_diff = []
                    for j in range(num_inter):
                        if v[j] != v_[j]:
                            list_diff.append(j)
                    out_str = '当前ms={:d}，活动={:s}，不同的列数为='.format(i + 1, k)
                    # print(out_str + str(list_diff))
                    pass
                else:
                    pass
            a_gx.append(a)
        sum = np.array(len(sum_gx))
        a_s = np.array(a_gx[:-1])
        print(sum)
        print(a_s / sum)
        print(a_gx)


        # Traverse all the diagrams and compare the adjacent ones
        for i in range(len(list_graph_dict)-1):
                j = i+1
                # Get the information for the Figure
                dict1 = list_graph_dict[i]
                dict2 = list_graph_dict[j]
                by_key1 = sorted(dict1.items(), key=lambda item: item[0])
                by_key2 = sorted(dict2.items(), key=lambda item: item[0])
                index1 = 0
                index2 = 0
                flag1 = True
                flag2 = True
                gz = gz_list[0][i]
                dic = gz_list[1][i]
                while(index1 < len(by_key1) and index2 < len(by_key2)):
                    if index1 == len(by_key1):
                        flag1 = False
                        index1 = index1 - 1
                    if index2 == len(by_key2):
                        flag2 = False
                        index2 = index2 - 1
                    if not flag1:
                        for i_index in range(index2, len(by_key2)):
                            if key not in dic:
                                next_code = get_next_code(dic)
                                gz.node(next_code, key, style='filled', color='green')
                                dic[hou_node_2] = next_code
                            else:
                                pass

                            # It is shown here that the nodes of the graph are already compared, and there is still work to be done in the previous figure
                            node2 = by_key2[i_index]  # node2为结点信息，[节点名，后继，前驱]
                            for hou_node in node2[1][0]:
                                if hou_node not in dic:
                                    next_code = get_next_code(dic)
                                    gz.node(next_code, key, style='filled', color='green')
                                    dic[hou_node] = next_code
                                gz.edge(dic[node1[0]], dic[hou_node], color='green')
                        index2 += 1
                        break
                    if not flag2:
                        for i_index in range(index1, len(by_key1)):
                            key = by_key2[i_index][0]
                            node1 = by_key1[i_index]
                            for hou_node in node1[1][0]:
                                gz.edge(dic[node1[0]], dic[hou_node], color='yellow')
                        index1 += 1
                        break

                    # Node compares the current node names as well as successors and forwards
                    node1 = by_key1[index1]
                    node2 = by_key2[index2]
                    if node1[0] == node2[0]:
                        # The node name is the same, comparing successors and forwards
                        if node1[1] == node2[1]:
                            pass
                        else:
                            draw_edges(node1, node2, gz_list, i)
                            key = by_key2[index2][0]
                        index1 += 1
                        index2 += 1
                    elif node1[0] > node2[0]:
                        key = by_key2[index2][0]
                        if key not in dic:
                            next_code = get_next_code(dic)
                            gz.node(next_code, key, style='filled', color='purple')
                            dic[key] = next_code
                        else:
                            pass
                            # Explain here: (1) The nodes in the post-figure are not in the previous figure, and all the lines in the following nodes are marked
                            # (2) Mark all the connections behind the wires in the previous figure
                            # (3) node2 is node information: node name, successor, forward
                            # Traverse the Subsequent
                        for hou_node in node2[1][0]:
                            if hou_node not in dic:
                                next_code = get_next_code(dic)
                                gz.node(next_code, key, style='filled', color='purple')
                                dic[hou_node] = next_code
                            gz.edge(dic[node2[0]], dic[hou_node], color='purple')
                        index2 += 1
                        pass
                    else:
                        index1 += 1
                        key = by_key1[index1][0]
                        for hou_node in node1[1][0]:
                            gz.edge(dic[node1[0]], dic[hou_node], color='red')
        # The difference graph output
        for gz, dic in zip(gz_list[0], gz_list[1]):
            gz. view()




if __name__ == '__main__':
    main()
